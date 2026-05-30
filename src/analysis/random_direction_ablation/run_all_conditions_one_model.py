"""
Single-process driver: load one aligned model once, run all 7 ablation
conditions on the same prompt set, write outputs as one bundle per
(condition, domain). Saves 6× model-load cost vs. spawning a fresh process
per condition.

Conditions (in order):
  1. baseline                     — no intervention; un-ablated aligned-model generations
  2. v_cross_L{L*}_ablated        — PASTA condition (ablate the v_align[L*] direction)
  3..7. v_random_k{k}_ablated     — random orthogonal directions, k∈{0..4}

The condition tag "baseline" is the on-disk identifier for the
no-intervention aligned-model output (matches the dir name
`strategy_A_naturalistic` in the generation tree).

Decoding: same per-sample seed (42 + sample_id) across all 7 conditions, so
any difference reflects only the choice of direction.

Bundle schema mirrors `data/generations/pasta_ablated/`. The aligned (instruct)
side is the only target; the base side is not used for the random-direction
control.

Usage (typically called by run_all_4gpu.py, not by hand):
  CUDA_VISIBLE_DEVICES=0 python run_all_conditions_one_model.py \\
      --config configs/config_gemma_2_9b.yaml --device cuda:0
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import yaml
from tqdm import tqdm

_HERE = Path(__file__).resolve().parent
_REPO_SRC = _HERE.parents[2] / "src"
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_REPO_SRC))

from _batched_generation import chat_prompt_ids_batch, generate_once_batch

from pasta.config import load_config
from pasta.generation import build_generation_record
from pasta.hooks import build_all_ablation_hooks
from pasta.io import write_record_incremental
from pasta.models import load_model
from pasta.prompts import build_strategy_A_prompt, load_prefixes


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--device", default=None)
    ap.add_argument("--skip-existing", action="store_true",
                    help="Skip a (condition, domain) cell if its output already "
                         "has the expected number of generations.")
    ap.add_argument("--domains", nargs="+", default=None)
    ap.add_argument("--conditions", nargs="+", default=None,
                    help="Subset of condition tags to run "
                         "(e.g. baseline v_cross_L2_ablated v_random_k0_ablated)")
    ap.add_argument("--batch-size", type=int, default=8,
                    help="Prompts per generate() call. H100s can typically "
                         "handle 16-32; 8 is a safe default.")
    return ap.parse_args()


def conditions_for(cfg: dict) -> list[dict]:
    rd = cfg["random_direction"]
    L_star = int(rd["best_layer"])
    v_align = rd["v_align_path"]
    dirs_root = Path(cfg["output"]["directions_subdir"])

    conds = [
        {"tag": "baseline", "intervention": "none", "direction_path": None},
        {"tag": f"v_cross_L{L_star}_ablated", "intervention": "ablate",
         "direction_path": v_align},
    ]
    for k in rd["seeds"]:
        conds.append({
            "tag": f"v_random_k{k}_ablated", "intervention": "ablate",
            "direction_path": str(dirs_root / f"v_random_k{k}.pt"),
        })
    return conds


def strategy_dir_name(baseline: str, tag: str) -> str:
    return baseline if tag == "baseline" else f"{baseline}_{tag}"


def out_dir_for(cfg: dict, tag: str) -> Path:
    base = cfg["output"]["baseline_strategy_dir"]
    strategy = strategy_dir_name(base, tag)
    return Path(cfg["output"]["generations_subdir"]) / strategy / "instruct"


def cell_complete(out_dir: Path, domain: str, expected_n: int) -> bool:
    p = out_dir / f"{domain}.json"
    if not p.exists():
        return False
    try:
        import json
        d = json.load(open(p))
    except Exception:
        return False
    return d.get("num_generations", 0) >= expected_n


def main():
    args = parse_args()
    cfg = load_config(Path(args.config))
    domains = args.domains or cfg["domains"]
    n_per_domain = int(cfg["n_prefixes_per_domain"])
    seed_choice = int(cfg["prefix_selection_seed"])
    base_seed = int(cfg["generation"]["seed"])
    gen_cfg = cfg["generation"]

    conds = conditions_for(cfg)
    if args.conditions:
        wanted = set(args.conditions)
        conds = [c for c in conds if c["tag"] in wanted]

    prompts_subdir = cfg.get("prompts_subdir")
    domain_to_prefixes: dict[str, list[dict]] = {}
    for d in domains:
        all_pre = load_prefixes(d, override_subdir=prompts_subdir)
        chosen = sorted([p for p in all_pre if p["id"] < n_per_domain],
                        key=lambda p: p["id"])
        domain_to_prefixes[d] = chosen
        chosen_ids = [p["id"] for p in chosen]
        print(f"[prompts] {d}: ids={chosen_ids[:3]}…{chosen_ids[-1]} (n={len(chosen_ids)})")

    pending = []
    for c in conds:
        odir = out_dir_for(cfg, c["tag"])
        cell_status = []
        for d in domains:
            ok = cell_complete(odir, d, n_per_domain)
            cell_status.append((d, ok))
        all_ok = all(ok for _, ok in cell_status)
        if args.skip_existing and all_ok:
            print(f"[skip] {c['tag']}: complete at {odir}")
            continue
        pending.append((c, odir, cell_status))

    if not pending:
        print("[done] nothing to do (all conditions complete)")
        return 0

    model_id = cfg["models"]["instruct"]
    device = args.device or cfg["devices"]["instruct"]
    print(f"\n[load] {model_id} on {device}")
    loaded = load_model(model_id, device)

    for c, odir, cell_status in pending:
        print(f"\n=== condition: {c['tag']}  →  {odir} ===")
        if c["intervention"] == "none":
            fwd_pre_hooks, fwd_hooks = [], []
        else:
            direction = torch.load(c["direction_path"], map_location="cpu",
                                   weights_only=False).float()
            n = float(direction.norm())
            assert abs(n - 1.0) < 1e-3, f"non-unit direction at {c['direction_path']}: ‖·‖={n}"
            fwd_pre_hooks, fwd_hooks = build_all_ablation_hooks(loaded, direction)
            print(f"[hooks] ablate {c['direction_path']}  ‖·‖={n:.4f}  "
                  f"sites=block_in×{loaded.n_layers} + attn_out×{loaded.n_layers} + mlp_out×{loaded.n_layers}")

        bundle_meta_template = {
            "model_type": "instruct",
            "model_id": model_id,
            "strategy_id": "A",
            "strategy_name": "naturalistic",
            "quantization": "bf16",
            "generation_params": {
                "temperature": gen_cfg["temperature"],
                "top_p": gen_cfg["top_p"],
                "max_new_tokens": gen_cfg["max_new_tokens"],
            },
            "intervention": {
                "mode": c["intervention"],
                "direction_path": c["direction_path"],
                "tag": c["tag"],
            },
        }

        for d in domains:
            cell_path = odir / f"{d}.json"
            if args.skip_existing and cell_complete(odir, d, n_per_domain):
                print(f"  [skip] {d} already complete")
                continue
            if cell_path.exists():
                stale = cell_path.with_suffix(".json.stale")
                cell_path.rename(stale)
                print(f"  [rotate] {cell_path} → {stale}")

            bundle_meta = {**bundle_meta_template, "domain": d}
            entries = domain_to_prefixes[d]
            B = max(1, int(args.batch_size))
            n = len(entries)
            pbar = tqdm(range(0, n, B), desc=f"{c['tag']}/{d}", leave=False)
            for batch_start in pbar:
                batch = entries[batch_start: batch_start + B]
                batch_messages = [build_strategy_A_prompt(d, e)["aligned_messages"] for e in batch]
                input_ids, attention_mask = chat_prompt_ids_batch(loaded, batch_messages)
                results = generate_once_batch(
                    loaded=loaded,
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=gen_cfg["max_new_tokens"],
                    temperature=gen_cfg["temperature"],
                    top_p=gen_cfg["top_p"],
                    fwd_pre_hooks=fwd_pre_hooks,
                    fwd_hooks=fwd_hooks,
                    seed=base_seed + batch_start,
                )
                for offset, (entry, result) in enumerate(zip(batch, results)):
                    sample_id = batch_start + offset
                    rec = build_generation_record(
                        strategy_id="A", domain=d, model_type="instruct",
                        model_id=model_id, sample_id=sample_id,
                        original_prefix_id=entry["id"],
                        prompt_used=batch_messages[offset],
                        raw_text=result["raw_text"],
                        finish_reason=result["finish_reason"],
                        usage=result["usage"],
                    )
                    write_record_incremental(odir, d, rec, bundle_meta)

    print("\n[done] all conditions complete for", cfg["pair"]["id"])
    return 0


if __name__ == "__main__":
    sys.exit(main())

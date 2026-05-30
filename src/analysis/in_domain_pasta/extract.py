"""Per-domain cross-model direction extraction (baseline generation + extraction).

This module merges the two former scripts (`baseline_gen.py` +
`extract_directions.py`) into one CLI with two phases that share the same
config, model spec, and train-prompt selection.

Phase 1 — baseline (no-intervention generation):
  For each domain, generate the ALIGNED and BASE model outputs on that domain's
  TRAIN prompts only. Train ids come from `select_train_prefix_ids` (Path A
  canonical: hash-seeded sample of `n_train_per_domain` ids drawn from the
  held-out pool, disjoint from the `[0..n_test-1]` test set). These generations
  are the teacher-forcing inputs for phase 2.

  Output bundle per side/domain:
    data/analysis_results/in_domain_pasta/inputs/baseline_generations/<alias>/<aligned|base>/<domain>.json

Phase 2 — extract (per-domain direction):
  For each domain:
    1. Teacher-force the aligned model on (its prompt + its aligned baseline gen)
       for every train prompt; accumulate the per-layer residual-stream sum over
       the generation positions -> mean_aligned_D[L].
    2. Teacher-force the base model on (its prompt + its base baseline gen) the
       same way -> mean_base_D[L].
    3. v_cross_D[L] = unit(mean_aligned_D[L] - mean_base_D[L]) for every L.

  The teacher-forcing loop is NOT reimplemented here: it is `pasta.extraction.
  accumulate_means`. Directions are saved per domain, matching the layout the
  downstream ablation scripts already read:
    data/analysis_results/in_domain_pasta/directions/<alias>/<domain>/v_cross_L<L>.pt

The two model copies are pinned to `--gpu-aligned` / `--gpu-base`; set them equal
for sequential single-GPU mode. In phase 1 each side is generated on its own GPU.

Usage:
  python extract.py --model llama-3.1-8b-instruct --phase all --gpu-aligned 0 --gpu-base 1
  python extract.py --model llama-3.1-8b-instruct --phase baseline --gpu-aligned 0 --gpu-base 1
  python extract.py --model llama-3.1-8b-instruct --phase extract  --gpu-aligned 0 --gpu-base 1
  python extract.py --model qwen2.5-7b-instruct   --phase all       --gpu-aligned 0 --gpu-base 0
  python extract.py --model gemma-2-9b-it --phase all --skip-existing
  python extract.py --model gemma-2-9b-it --phase all --dry-run
"""

from __future__ import annotations

import argparse
import sys
import threading
import time
from pathlib import Path

import torch
import yaml

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[2]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(REPO_ROOT / "src"))

from _domain_selection import select_train_prefix_ids

from pasta.extraction import accumulate_means
from pasta.generation import (
    build_generation_record,
    chat_prompt_ids,
    generate_once,
    raw_prompt_ids,
)
from pasta.io import read_json, write_record_incremental
from pasta.models import load_model
from pasta.prompts import build_strategy_A_prompt, load_prefixes

DATA_ROOT = REPO_ROOT / "data" / "analysis_results" / "in_domain_pasta"
BASELINE_ROOT = DATA_ROOT / "inputs" / "baseline_generations"
DIRECTIONS_ROOT = DATA_ROOT / "directions"

TARGETS = ("aligned", "base")


def parse_args() -> argparse.Namespace:
    """CLI: one model alias, one phase, and the two GPU pins."""
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--config", default=str(HERE / "configs" / "models.yaml"))
    p.add_argument(
        "--model",
        required=True,
        help="Model alias (must match one of `models:` in --config).",
    )
    p.add_argument(
        "--phase",
        choices=["baseline", "extract", "all"],
        default="all",
        help="baseline = generate train outputs; extract = build directions; all = both.",
    )
    p.add_argument(
        "--gpu-aligned", type=int, default=0, help="GPU id for the aligned model."
    )
    p.add_argument(
        "--gpu-base",
        type=int,
        default=1,
        help="GPU id for the base model. Set equal to --gpu-aligned for sequential single-GPU mode.",
    )
    p.add_argument(
        "--domains",
        default=None,
        help="Comma-separated domain shard (default: all in config).",
    )
    p.add_argument(
        "--skip-existing",
        action="store_true",
        help="Phase baseline: skip train ids already present in the output bundle.",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Phase extract: teacher-force one record per (model, domain) for shape checks.",
    )
    return p.parse_args()


def pick_model(cfg: dict, alias: str) -> dict:
    """Select the single `models:` entry matching `alias` or exit."""
    spec = next((m for m in cfg["models"] if m["alias"] == alias), None)
    if spec is None:
        raise SystemExit(
            f"unknown model alias '{alias}'. "
            f"Known: {[m['alias'] for m in cfg['models']]}"
        )
    return spec


def resolve_domains(cfg: dict, arg: str | None) -> list[str]:
    """All config domains, or the comma-separated shard from --domains."""
    if arg is None:
        return list(cfg["domains"])
    return [d.strip() for d in arg.split(",")]


def load_baseline_records(baseline_dir: Path, target: str, domain: str) -> list[dict]:
    """Read one side's per-domain baseline bundle into accumulate_means records.

    Returns dicts carrying domain / sample_id / raw_text / prompt_used, matching
    the fields `pasta.extraction.accumulate_means` consumes.
    """
    fp = baseline_dir / target / f"{domain}.json"
    if not fp.exists():
        raise SystemExit(
            f"missing baseline generations: {fp}. Run --phase baseline first."
        )
    bundle = read_json(fp)
    return [
        {
            "domain": domain,
            "sample_id": g["sample_id"],
            "raw_text": g["raw_text"],
            "prompt_used": g["prompt_used"],
        }
        for g in bundle["generations"]
    ]


def generate_side(
    spec: dict,
    target: str,
    domains: list[str],
    cfg: dict,
    device: str,
    skip_existing: bool,
) -> int:
    """Generate one side's train-prompt baseline outputs for every domain.

    Reuses pasta.generation (generate_once + chat/raw prompt ids +
    build_generation_record) and pasta.io.write_record_incremental, writing the
    same bundle layout the former baseline_gen.py produced.
    """
    gen_cfg = cfg["generation"]
    n_train = cfg["n_train_per_domain"]
    prompts_subdir = cfg.get("prompts_subdir")
    model_id = spec["id"] if target == "aligned" else spec["base_id"]
    out_root = BASELINE_ROOT / spec["alias"] / target

    print(f"[baseline/{target}] model={model_id} -> {out_root}", flush=True)
    t_load = time.time()
    loaded = load_model(model_id, device=device)
    print(
        f"[baseline/{target}] loaded in {time.time() - t_load:.1f}s "
        f"(n_layers={loaded.n_layers}, d_model={loaded.d_model})",
        flush=True,
    )

    bundle_meta_template = {
        "model_type": target,
        "model_id": model_id,
        "strategy_id": "A",
        "strategy_name": "naturalistic",
        "quantization": "bf16",
        "generation_params": {
            "temperature": gen_cfg["temperature"],
            "top_p": gen_cfg["top_p"],
            "max_new_tokens": gen_cfg["max_new_tokens"],
        },
        "intervention": {"mode": "none", "tag": "baseline_train"},
    }

    written = 0
    for domain in domains:
        train_ids = select_train_prefix_ids(domain, n_train, prompts_subdir=prompts_subdir)
        by_id = {p["id"]: p for p in load_prefixes(domain, override_subdir=prompts_subdir)}
        entries = [by_id[i] for i in train_ids]

        out_fp = out_root / f"{domain}.json"
        done_ids: set = set()
        if skip_existing and out_fp.exists():
            try:
                done_ids = {
                    g["original_prefix_id"]
                    for g in read_json(out_fp).get("generations", [])
                }
            except Exception:
                pass
        missing = [e for e in entries if e["id"] not in done_ids]
        if not missing:
            print(f"[baseline/{target}/{domain}] {len(entries)} done — skip", flush=True)
            continue

        print(
            f"[baseline/{target}/{domain}] train_ids={train_ids[:5]}…(n={len(train_ids)})  "
            f"todo={len(missing)}",
            flush=True,
        )
        t_dom = time.time()
        bundle_meta = {**bundle_meta_template, "domain": domain}
        for entry in missing:
            prompts = build_strategy_A_prompt(domain, entry)
            if target == "aligned":
                input_ids = chat_prompt_ids(loaded, prompts["aligned_messages"])
                prompt_used = prompts["aligned_messages"]
            else:
                input_ids = raw_prompt_ids(loaded, prompts["base_prompt"])
                prompt_used = prompts["base_prompt"]
            try:
                result = generate_once(
                    loaded=loaded,
                    input_ids=input_ids,
                    max_new_tokens=gen_cfg["max_new_tokens"],
                    temperature=gen_cfg["temperature"],
                    top_p=gen_cfg["top_p"],
                    seed=gen_cfg["seed"] + entry["id"],
                )
            except Exception as e:
                print(f"  ERROR id={entry['id']}: {type(e).__name__}: {e}", flush=True)
                continue
            record = build_generation_record(
                strategy_id="A",
                domain=domain,
                model_type=target,
                model_id=model_id,
                sample_id=entry["id"],
                original_prefix_id=entry["id"],
                prompt_used=prompt_used,
                raw_text=result["raw_text"],
                finish_reason=result["finish_reason"],
                usage=result["usage"],
            )
            write_record_incremental(out_root, domain, record, bundle_meta)
            written += 1
        print(
            f"  [baseline/{target}/{domain}] done {len(missing)} in {time.time() - t_dom:.0f}s",
            flush=True,
        )

    del loaded
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return written


def run_baseline(spec: dict, domains: list[str], cfg: dict, args: argparse.Namespace) -> None:
    """Generate both sides' baseline outputs, parallel across GPUs if distinct."""
    parallel = args.gpu_aligned != args.gpu_base
    print(
        f"[baseline] aligned on cuda:{args.gpu_aligned}  base on cuda:{args.gpu_base}  "
        f"parallel={parallel}",
        flush=True,
    )
    t_global = time.time()
    if parallel:
        totals: dict[str, int] = {}

        def work(target: str, gpu: int) -> None:
            totals[target] = generate_side(
                spec, target, domains, cfg, f"cuda:{gpu}", args.skip_existing
            )

        ta = threading.Thread(target=work, args=("aligned", args.gpu_aligned))
        tb = threading.Thread(target=work, args=("base", args.gpu_base))
        ta.start()
        tb.start()
        ta.join()
        tb.join()
        grand_total = sum(totals.values())
    else:
        grand_total = sum(
            generate_side(spec, target, domains, cfg, f"cuda:{args.gpu_aligned}", args.skip_existing)
            for target in TARGETS
        )
    print(f"\n[baseline done] total={grand_total} in {time.time() - t_global:.0f}s", flush=True)


def write_domain_directions(
    out_dir: Path,
    sum_a: torch.Tensor,
    n_a: int,
    sum_b: torch.Tensor,
    n_b: int,
    n_layers: int,
) -> None:
    """v_cross[L] = unit(mean_aligned[L] - mean_base[L]); save one tensor per layer.

    Layout matches the former extract_directions.py (per-domain v_cross_L<L>.pt).
    """
    mean_a = sum_a / n_a
    mean_b = sum_b / n_b
    diff = mean_a - mean_b
    out_dir.mkdir(parents=True, exist_ok=True)
    for L in range(n_layers):
        v = diff[L].float()
        v_unit = v / (v.norm() + 1e-8)
        torch.save(v_unit, out_dir / f"v_cross_L{L}.pt")


def run_extract(spec: dict, domains: list[str], args: argparse.Namespace) -> None:
    """Teacher-force both sides per domain and write v_cross directions.

    The accumulation loop is `pasta.extraction.accumulate_means` (not
    reimplemented here).
    """
    baseline_dir = BASELINE_ROOT / spec["alias"]
    out_dir_root = DIRECTIONS_ROOT / spec["alias"]
    parallel = args.gpu_aligned != args.gpu_base
    print(
        f"[extract] aligned on cuda:{args.gpu_aligned}  base on cuda:{args.gpu_base}  "
        f"parallel={parallel}",
        flush=True,
    )

    print("[extract] loading aligned…", flush=True)
    t_a = time.time()
    aligned = load_model(spec["id"], device=f"cuda:{args.gpu_aligned}")
    print(f"  loaded aligned in {time.time() - t_a:.1f}s", flush=True)

    if parallel:
        print("[extract] loading base in parallel…", flush=True)
        t_b = time.time()
        base = load_model(spec["base_id"], device=f"cuda:{args.gpu_base}")
        print(f"  loaded base in {time.time() - t_b:.1f}s", flush=True)
    else:
        base = None

    n_layers = aligned.n_layers
    d_model = aligned.d_model

    t_global = time.time()
    for domain in domains:
        print(f"\n[extract/{domain}] reading baseline records…", flush=True)
        recs_aligned = load_baseline_records(baseline_dir, "aligned", domain)
        recs_base = load_baseline_records(baseline_dir, "base", domain)
        if args.dry_run:
            recs_aligned = recs_aligned[:1]
            recs_base = recs_base[:1]
        print(f"  aligned={len(recs_aligned)} base={len(recs_base)}", flush=True)

        if parallel:
            results: dict[str, tuple[torch.Tensor, int]] = {}

            def work_a() -> None:
                results["aligned"] = accumulate_means(
                    aligned, recs_aligned, n_layers, d_model, f"{domain}-aligned"
                )

            def work_b() -> None:
                results["base"] = accumulate_means(
                    base, recs_base, n_layers, d_model, f"{domain}-base"
                )

            ta = threading.Thread(target=work_a)
            tb = threading.Thread(target=work_b)
            ta.start()
            tb.start()
            ta.join()
            tb.join()
            sum_a, n_a = results["aligned"]
            sum_b, n_b = results["base"]
        else:
            sum_a, n_a = accumulate_means(
                aligned, recs_aligned, n_layers, d_model, f"{domain}-aligned"
            )
            if base is None:
                del aligned
                torch.cuda.empty_cache()
                base = load_model(spec["base_id"], device=f"cuda:{args.gpu_base}")
            sum_b, n_b = accumulate_means(
                base, recs_base, n_layers, d_model, f"{domain}-base"
            )

        if args.dry_run:
            print(
                f"[dry-run/{domain}] aligned positions: {n_a} base positions: {n_b}",
                flush=True,
            )
            continue

        out_dir = out_dir_root / domain
        write_domain_directions(
            out_dir, sum_a, n_a, sum_b, n_b, n_layers,
        )
        print(
            f"[extract/{domain}] wrote {n_layers} directions  "
            f"aligned_pos={n_a} base_pos={n_b}",
            flush=True,
        )

    print(f"\n[extract done] in {time.time() - t_global:.0f}s", flush=True)


def main() -> None:
    args = parse_args()
    cfg = yaml.safe_load(open(args.config))
    spec = pick_model(cfg, args.model)
    domains = resolve_domains(cfg, args.domains)

    if args.phase in ("baseline", "all"):
        run_baseline(spec, domains, cfg, args)
    if args.phase in ("extract", "all"):
        run_extract(spec, domains, args)


if __name__ == "__main__":
    main()

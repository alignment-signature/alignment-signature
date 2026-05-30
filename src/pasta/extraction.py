"""Phase A — per-layer cross-model direction extraction.

For each of Base and Aligned:
  - Load the Strategy-A baseline generation records that the model itself produced.
  - Teacher-force each model on (its prompt ids + its generation ids) — single forward pass.
  - At every layer, take the residual-stream slice spanning the generation
    positions and accumulate a running sum + position count.

Then per layer:
    v_cross[L] = unit_normalize( mean_aligned[L] - mean_base[L] )

These directions are written to `outputs/directions/v_cross_L{L}.pt` (or wherever
`cfg.output.directions_subdir` points), one tensor per layer.
"""

from __future__ import annotations

from pathlib import Path

import torch
from tqdm import tqdm

from pasta.generation import chat_prompt_ids, raw_prompt_ids
from pasta.io import read_json, write_json
from pasta.models import LoadedModel
from pasta.paths import baseline_dir_for


_REQUIRED_RECORD_FIELDS = ("sample_id", "raw_text", "prompt_used", "original_prefix_id")


def load_baseline_records(cfg: dict, model_type: str) -> list[dict]:
    """Load externally-produced generation records for one side, flattened across domains.

    Pasta does not generate this text. The caller writes per-domain bundles to:
      - `data/generations/aligned/<pair>/<t>/<domain>.json` for the instruct side
      - `data/generations/base/<pair>/<t>/<domain>.json` for the base side
    (paths are computed from `cfg.pair.id` and `cfg.output.temperature_tag`).

    Each bundle JSON must declare `model_type` and `model_id` matching the requested
    side and `cfg.models[<side>]` — this enforces "each model's own outputs". Each
    `generations[i]` entry must include `sample_id`, `raw_text`, `prompt_used`, and
    `original_prefix_id` (the source-prefix identifier used for cross-side matching).
    """
    expected_model_id = cfg["models"][model_type]
    root = baseline_dir_for(cfg, model_type)
    if not root.exists():
        raise SystemExit(
            f"Missing {root}. Pasta does not produce baseline generations; place the "
            f"{model_type} model's per-domain generation bundles under this directory "
            f"before running extraction."
        )
    records = []
    for domain in cfg["domains"]:
        path = root / f"{domain}.json"
        if not path.exists():
            raise SystemExit(f"Missing {path}")
        bundle = read_json(path)

        bundle_mt = bundle.get("model_type")
        bundle_mid = bundle.get("model_id")
        if bundle_mt != model_type:
            raise SystemExit(
                f"{path}: bundle.model_type={bundle_mt!r} does not match the side being "
                f"loaded ({model_type!r}). Records must be each model's own outputs."
            )
        if bundle_mid != expected_model_id:
            raise SystemExit(
                f"{path}: bundle.model_id={bundle_mid!r} does not match "
                f"cfg.models.{model_type}={expected_model_id!r}. Records must come from "
                f"the configured model."
            )

        for g in bundle["generations"]:
            missing = [f for f in _REQUIRED_RECORD_FIELDS if f not in g]
            if missing:
                raise SystemExit(
                    f"{path}: generation entry is missing required field(s) {missing}. "
                    f"Each record needs sample_id, raw_text, prompt_used, and "
                    f"original_prefix_id so cross-side prompt matching can be verified."
                )
            records.append({
                "domain": domain,
                "sample_id": g["sample_id"],
                "raw_text": g["raw_text"],
                "prompt_used": g["prompt_used"],
                "original_prefix_id": g["original_prefix_id"],
            })
    return records


def validate_paired_records(base_records: list[dict], inst_records: list[dict]) -> None:
    """Enforce that the two sides are matched outputs over matched prompts.

    Concretely:
      - The (domain, sample_id) sets agree exactly across sides.
      - For each (domain, sample_id), `original_prefix_id` is the same on both sides
        (i.e., both models actually saw the same source prefix).

    `prompt_used` is intentionally NOT compared: base models receive a raw prompt
    while instruct models receive chat-template messages, so the two shapes differ
    by design. `original_prefix_id` is the underlying invariant.
    """
    def index(records: list[dict]) -> dict:
        out: dict[tuple, dict] = {}
        for r in records:
            key = (r["domain"], r["sample_id"])
            if key in out:
                raise SystemExit(
                    f"Duplicate (domain, sample_id) {key} within a single side — each "
                    f"sample must appear at most once per side."
                )
            out[key] = r
        return out

    base_idx = index(base_records)
    inst_idx = index(inst_records)
    only_base = set(base_idx) - set(inst_idx)
    only_inst = set(inst_idx) - set(base_idx)
    if only_base or only_inst:
        raise SystemExit(
            "Paired-records contract violated: base and instruct must cover the "
            "same (domain, sample_id) set.\n"
            f"  only in base:     {sorted(only_base)[:10]}{' ...' if len(only_base) > 10 else ''}\n"
            f"  only in instruct: {sorted(only_inst)[:10]}{' ...' if len(only_inst) > 10 else ''}"
        )

    mismatched = [
        (key, base_idx[key]["original_prefix_id"], inst_idx[key]["original_prefix_id"])
        for key in base_idx
        if base_idx[key]["original_prefix_id"] != inst_idx[key]["original_prefix_id"]
    ]
    if mismatched:
        sample = mismatched[:5]
        raise SystemExit(
            "Paired-records contract violated: original_prefix_id differs across "
            "sides for matched (domain, sample_id) keys — the two models were not "
            "run on the same source prefixes.\n"
            + "\n".join(
                f"  {key}: base.original_prefix_id={b!r} instruct.original_prefix_id={i!r}"
                for key, b, i in sample
            )
            + (f"\n  ... and {len(mismatched) - len(sample)} more" if len(mismatched) > len(sample) else "")
        )


def _build_input_ids(loaded: LoadedModel, prompt_used) -> torch.Tensor:
    if isinstance(prompt_used, list):
        return chat_prompt_ids(loaded, prompt_used)
    return raw_prompt_ids(loaded, prompt_used)


def _encode_generation(loaded: LoadedModel, raw_text: str) -> torch.Tensor:
    ids = loaded.tokenizer.encode(raw_text, add_special_tokens=False)
    return torch.tensor([ids], device=loaded.device)


def accumulate_means(
    loaded: LoadedModel,
    records: list[dict],
    n_layers: int,
    d_model: int,
    label: str,
) -> tuple[torch.Tensor, int]:
    """Teacher-force each record, accumulate per-layer sum over generation positions."""
    running_sum = torch.zeros((n_layers, d_model), dtype=torch.float64)
    total_positions = 0

    for rec in tqdm(records, desc=label, leave=True):
        prompt_ids = _build_input_ids(loaded, rec["prompt_used"])
        gen_ids = _encode_generation(loaded, rec["raw_text"])
        full = torch.cat([prompt_ids, gen_ids], dim=1)

        prompt_len = prompt_ids.shape[1]
        gen_len = gen_ids.shape[1]
        if gen_len == 0:
            raise ValueError(
                f"No generation text to teacher-force for record "
                f"domain={rec['domain']!r} sample_id={rec['sample_id']!r}: "
                f"raw_text encoded to 0 tokens. Re-run baseline generation "
                f"so every record has non-empty raw_text."
            )
        start = prompt_len - 1
        end = start + gen_len

        with torch.no_grad():
            out = loaded.model(full, output_hidden_states=True, use_cache=False)

        for L in range(n_layers):
            h = out.hidden_states[L + 1][0, start:end, :]
            running_sum[L] += h.to(torch.float64).sum(dim=0).cpu()
        total_positions += gen_len

        del out

    return running_sum, total_positions


def write_directions(
    out_dir: Path,
    base_sum: torch.Tensor,
    base_n: int,
    inst_sum: torch.Tensor,
    inst_n: int,
    n_layers: int,
    d_model: int,
    base_records: int,
    inst_records: int,
) -> None:
    """Compute v_cross[L] = unit(mean_aligned[L] - mean_base[L]) and persist all L tensors."""
    base_mean = base_sum / base_n
    inst_mean = inst_sum / inst_n
    diff = inst_mean - base_mean
    norms = diff.norm(dim=-1)
    print("\n[diff] per-layer ||v_cross_unnorm||:")
    for L in range(n_layers):
        print(f"  L{L:2d}: {norms[L].item():.3f}")

    out_dir.mkdir(parents=True, exist_ok=True)
    for L in range(n_layers):
        v = diff[L].float()
        v_unit = v / (v.norm() + 1e-8)
        torch.save(v_unit, out_dir / f"v_cross_L{L}.pt")

    stats = {
        "n_layers": n_layers,
        "d_model": d_model,
        "base_generation_positions": base_n,
        "instruct_generation_positions": inst_n,
        "base_records": base_records,
        "instruct_records": inst_records,
        "per_layer_unnorm_norm": [norms[L].item() for L in range(n_layers)],
    }
    write_json(out_dir / "v_cross_stats.json", stats)
    print(f"\n[done] wrote {n_layers} directions + v_cross_stats.json to {out_dir}")

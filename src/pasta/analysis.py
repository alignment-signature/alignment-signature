"""Aggregate per-layer Pangram results into a single-pair RESULTS.md.

Reads Pangram detection outputs that mirror the generations layout under
`<pangram_root>/`, computes the aligned + base baselines, per-layer mean
fraction_ai, and ΔPangram vs the aligned baseline. The best layer (lowest
fraction_ai) is highlighted.

NOTE — single-detector pipeline. This module hardcodes Pangram as the
reference detector: `cfg.output.pangram_root`, `_mean_fraction_ai` on the
`detection.fraction_ai` field, and the best-layer = argmin(fraction_ai)
selection are all Pangram-specific. The same selection logic in
`scripts/generate_best_layer.py:_find_best_layer` mirrors it. To support a
second detector, generalize the path lookup (read from a `<detector>_root`
config key), the field name, and update `scripts/run_pipeline.stage_sweep_pangram`
to invoke the corresponding evaluator.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

from pasta.io import read_json
from pasta.paths import (
    aligned_baseline_dir,
    base_baseline_dir,
    pangram_mirror,
    pasta_ablated_dir,
)


def _mean_fraction_ai(domain_dir: Path, domains: list[str]) -> tuple[float | None, int, int]:
    if not domain_dir.exists():
        return None, 0, 0
    vals: list[float] = []
    errors = 0
    for d in domains:
        p = domain_dir / f"{d}.json"
        if not p.exists():
            continue
        data = read_json(p)
        for r in data["results"]:
            det = r.get("detection")
            if isinstance(det, dict):
                if det.get("success"):
                    vals.append(det["fraction_ai"])
                else:
                    errors += 1
    if not vals:
        return None, 0, errors
    return sum(vals) / len(vals), len(vals), errors


def summarize(cfg: dict) -> dict:
    domains = cfg["domains"]
    n_layers = cfg["phase_A"]["n_layers"]

    aligned_pang = pangram_mirror(cfg, aligned_baseline_dir(cfg))
    base_pang = pangram_mirror(cfg, base_baseline_dir(cfg))
    aligned_mean, aligned_n, aligned_err = _mean_fraction_ai(aligned_pang, domains)
    base_mean, base_n, base_err = _mean_fraction_ai(base_pang, domains)

    per_layer = []
    for L in range(n_layers):
        gen_dir = pasta_ablated_dir(cfg, "instruct", f"v_cross_L{L}_ablated")
        ldir = pangram_mirror(cfg, gen_dir)
        m, n, e = _mean_fraction_ai(ldir, domains)
        per_layer.append({"layer": L, "mean_fraction_ai": m, "n": n, "errors": e})

    complete = [layer for layer in per_layer if layer["mean_fraction_ai"] is not None]
    best = min(complete, key=lambda r: r["mean_fraction_ai"]) if complete else None

    return {
        "pair_id": cfg.get("pair", {}).get("id", "single_pair"),
        "base_model": cfg["models"]["base"],
        "aligned_model": cfg["models"]["instruct"],
        "n_layers": n_layers,
        "baseline_aligned": aligned_mean,
        "baseline_aligned_n": aligned_n,
        "baseline_aligned_err": aligned_err,
        "baseline_base": base_mean,
        "baseline_base_n": base_n,
        "baseline_base_err": base_err,
        "per_layer": per_layer,
        "n_layers_done": len(complete),
        "best_layer": best,
    }


def _fmt(x):
    return "—" if x is None else f"{x:.3f}"


def render_table(row: dict) -> str:
    lines = [
        f"### {row['pair_id']}\n",
        f"- base: `{row['base_model']}`",
        f"- aligned: `{row['aligned_model']}`",
        f"- aligned baseline mean_fraction_ai: **{_fmt(row['baseline_aligned'])}** "
        f"(n={row['baseline_aligned_n']}, err={row['baseline_aligned_err']})",
        f"- base baseline mean_fraction_ai: **{_fmt(row['baseline_base'])}** "
        f"(n={row['baseline_base_n']}, err={row['baseline_base_err']})",
    ]
    if row["best_layer"]:
        b = row["best_layer"]
        delta = (b["mean_fraction_ai"] - row["baseline_aligned"]) if row["baseline_aligned"] is not None else None
        lines.append(
            f"- best layer: **L={b['layer']}** at mean_fraction_ai={_fmt(b['mean_fraction_ai'])} "
            f"(ΔPangram={('+' if delta is not None and delta > 0 else '')}{_fmt(delta)})"
        )
    lines.append("")
    lines.append("| Layer | mean_fraction_ai | n | errors | ΔPangram |")
    lines.append("|---:|---:|---:|---:|---:|")
    for layer in row["per_layer"]:
        delta = (
            (layer["mean_fraction_ai"] - row["baseline_aligned"])
            if (layer["mean_fraction_ai"] is not None and row["baseline_aligned"] is not None)
            else None
        )
        lines.append(
            f"| {layer['layer']} | {_fmt(layer['mean_fraction_ai'])} | {layer['n']} | {layer['errors']} | "
            f"{('+' if delta is not None and delta > 0 else '')}{_fmt(delta)} |"
        )
    return "\n".join(lines)


def render_results_md(row: dict) -> str:
    body = [
        f"# Alignment Removal — {row['pair_id']}\n",
        f"_Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M')}_\n",
        "Per-layer Pangram fraction_ai after ablating v_cross[L] at every residual-stream\n"
        "site of the aligned model. Aligned baseline = mean fraction_ai of the un-intervened\n"
        "aligned model. ΔPangram = layer fraction_ai − aligned baseline. More negative = stronger.\n",
        render_table(row),
    ]
    return "\n".join(body)

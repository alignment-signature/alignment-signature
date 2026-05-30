"""
Algorithm 1 — Sample random directions orthogonal to v_align[L*].

Per pre-registered seed k, draw u ~ N(0, I_d), Gram-Schmidt against
v_align[L*], renormalise. Save unit vectors to disk and emit verification
JSON (Step 2 of Algorithm 2: pre-flight checks).

Usage:
  uv run python sample_random_directions.py --config config_gemma_2_9b.yaml
  uv run python sample_random_directions.py --config <cfg> --extend  # add seeds {5..9}

The directions land at:
  <directions_subdir>/v_random_k{k}.pt          # unit-norm float32, shape (d,)
  <directions_subdir>/manifest.json             # provenance + pre-flight stats

This script is deterministic: identical seed → identical v_k.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
import yaml

MODULE_ROOT = Path(__file__).resolve().parent


def sample_random_orthogonal(v_align: torch.Tensor, k: int) -> torch.Tensor:
    """Algorithm 1. v_align must be a unit-norm 1-D tensor in float32.

    Returns a unit-norm vector orthogonal to v_align, deterministic in k.
    """
    assert v_align.ndim == 1, "v_align must be 1-D"
    assert v_align.dtype == torch.float32, "use float32 for stability"
    n = float(v_align.norm().item())
    assert abs(n - 1.0) < 1e-4, f"v_align must be unit-norm; got {n}"

    g = torch.Generator()
    g.manual_seed(int(k))
    d = v_align.shape[0]
    u = torch.randn(d, generator=g, dtype=torch.float32)

    coeff = torch.dot(u, v_align)
    u = u - coeff * v_align

    norm = u.norm()
    if norm.item() < 1e-8:
        raise RuntimeError(f"degenerate sample at seed={k}: residual norm {norm.item():.2e}")
    return u / norm


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--extend", action="store_true",
                    help="Generate the extension seed set (used only when "
                         "the pre-registered escape hatch fires).")
    ap.add_argument("--force", action="store_true",
                    help="Overwrite existing v_random_k*.pt files.")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    rd = cfg["random_direction"]

    v_align_path = Path(rd["v_align_path"])
    out_dir = Path(rd_out := cfg["output"]["directions_subdir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    seeds = rd["extension_seeds"] if args.extend else rd["seeds"]
    cos_thr = float(rd["pairwise_cos_threshold"])
    expected_d = int(rd["d_model"])

    v_align = torch.load(v_align_path, map_location="cpu", weights_only=False)
    if v_align.dtype != torch.float32:
        v_align = v_align.float()
    assert v_align.shape == (expected_d,), \
        f"v_align shape {tuple(v_align.shape)} != ({expected_d},)"

    v_align = v_align / v_align.norm()

    print(f"[sample_random] v_align: {v_align_path}")
    print(f"[sample_random] d={v_align.shape[0]}  ‖v_align‖={v_align.norm().item():.6f}")
    print(f"[sample_random] seeds={seeds}  output={out_dir}")

    sampled: dict[int, torch.Tensor] = {}
    for k in seeds:
        out_p = out_dir / f"v_random_k{k}.pt"
        if out_p.exists() and not args.force:
            print(f"[sample_random] k={k}: already exists at {out_p} (use --force to overwrite); reusing")
            v_k = torch.load(out_p, map_location="cpu", weights_only=False).float()
        else:
            v_k = sample_random_orthogonal(v_align, k)
            torch.save(v_k, out_p)
            print(f"[sample_random] k={k}: wrote {out_p}  ‖v_k‖={v_k.norm().item():.6f}  "
                  f"<v_k,v_align>={float(torch.dot(v_k, v_align)):.2e}")
        sampled[k] = v_k

    pre_flight = {"per_seed": [], "pairwise": [], "passed": True, "issues": []}
    for k, v in sampled.items():
        norm_dev = abs(v.norm().item() - 1.0)
        cos_align = abs(float(torch.dot(v, v_align)))
        pre_flight["per_seed"].append({
            "seed": int(k),
            "norm_minus_1": norm_dev,
            "cos_with_v_align": cos_align,
        })
        if norm_dev >= 1e-6:
            pre_flight["passed"] = False
            pre_flight["issues"].append(f"k={k}: |‖v‖-1| = {norm_dev:.2e} ≥ 1e-6")
        if cos_align >= 1e-6:
            pre_flight["passed"] = False
            pre_flight["issues"].append(f"k={k}: |<v,v_align>| = {cos_align:.2e} ≥ 1e-6")

    sk = sorted(sampled.keys())
    for i in range(len(sk)):
        for j in range(i + 1, len(sk)):
            kj, ki = sk[i], sk[j]
            cos_ij = abs(float(torch.dot(sampled[kj], sampled[ki])))
            pre_flight["pairwise"].append({
                "seed_a": int(kj), "seed_b": int(ki), "abs_cos": cos_ij,
            })
            if cos_ij >= cos_thr:
                pre_flight["passed"] = False
                pre_flight["issues"].append(
                    f"k={kj},k'={ki}: |cos| = {cos_ij:.4f} ≥ {cos_thr}"
                )

    manifest_path = out_dir / "manifest.json"
    manifest = {}
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
    manifest.setdefault("v_align_path", str(v_align_path))
    manifest.setdefault("d_model", int(expected_d))
    manifest.setdefault("seeds", [])
    for k in sampled:
        if k not in manifest["seeds"]:
            manifest["seeds"].append(int(k))
    manifest["pre_flight"] = pre_flight
    manifest_path.write_text(json.dumps(manifest, indent=2))

    print()
    print("[sample_random] pre-flight:", "PASS" if pre_flight["passed"] else "FAIL")
    for line in pre_flight["issues"]:
        print(f"  - {line}")
    if not pre_flight["passed"]:
        return 1
    print(f"[sample_random] manifest: {manifest_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

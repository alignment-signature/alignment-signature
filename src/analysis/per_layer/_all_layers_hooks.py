"""Build n_layers single-site hooks at once — one per layer, all active simultaneously.

The canonical PASTA pipeline registers 3 hooks at the model's best layer L*
only. This experiment registers ONE hook per layer (block-output) at *every*
layer L, each ablating that layer's own `v_cross[L]` direction. All hooks fire
in a single generation pass.

This is distinct from a per-layer SWEEP that registers one hook on one layer at
a time and runs n_layers separate generation passes.

Hypothesis: if the post-training alignment direction is distributed across
the depth of the model, ablating it at every layer's residual-stream output
simultaneously should drop the AI-detection score more than any single-layer
ablation does.
"""
from __future__ import annotations

from pathlib import Path
import torch

from pasta.hooks import get_direction_ablation_output_hook
from pasta.models import LoadedModel


def build_all_layers_single_site_hooks(loaded: LoadedModel, directions_dir: Path):
    """Register one forward post-hook on `blocks[L]` for every L in [0, n_layers),
    each ablating `directions_dir/v_cross_L<L>.pt` from that block's output.

    Returns (fwd_pre_hooks, fwd_hooks). fwd_pre_hooks is empty.
    """
    blocks = loaded.block_modules()
    fwd_post = []
    for L in range(loaded.n_layers):
        fp = directions_dir / f"v_cross_L{L}.pt"
        if not fp.exists():
            raise FileNotFoundError(f"missing direction tensor: {fp}")
        d = torch.load(fp, map_location="cpu")
        if d.dtype != torch.float32:
            d = d.float()
        fwd_post.append((blocks[L], get_direction_ablation_output_hook(d)))
    return [], fwd_post

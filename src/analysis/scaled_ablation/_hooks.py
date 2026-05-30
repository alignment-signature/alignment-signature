"""Scaled (alpha) direction-ablation hooks for the scaled-ablation experiment.

``pasta.hooks.build_all_ablation_hooks`` removes a direction fully (the alpha=1
case): ``h <- h - (h·d) d``. This analysis sweeps a multiplier ``alpha`` on that
projection at every block-input / attn-output / mlp-output site:

    h <- h - alpha * (h·d) d

alpha=0 is identity (aligned baseline), alpha=1 is full ablation (== pasta's),
alpha in (0,1) is partial, and alpha>1 over-ablates along the same direction.

This is the alpha-aware variant kept local to this analysis (the analysis
convention is to add hook variants here rather than touch ``pasta.hooks``, and
promote into ``pasta`` only when a second analysis needs it). Everything else
(model loading, generation, prompts, IO) is reused from ``pasta``.
"""
from __future__ import annotations

from typing import Callable

import torch

from pasta.models import LoadedModel


def _project_out(activation: torch.Tensor, direction: torch.Tensor, alpha: float) -> torch.Tensor:
    direction = direction / (direction.norm(dim=-1, keepdim=True) + 1e-8)
    direction = direction.to(activation)
    return activation - alpha * (activation @ direction).unsqueeze(-1) * direction


def _input_pre_hook(direction: torch.Tensor, alpha: float) -> Callable:
    def hook_fn(module, input):
        if isinstance(input, tuple):
            return (_project_out(input[0], direction, alpha), *input[1:])
        return _project_out(input, direction, alpha)
    return hook_fn


def _output_hook(direction: torch.Tensor, alpha: float) -> Callable:
    def hook_fn(module, input, output):
        if isinstance(output, tuple):
            return (_project_out(output[0], direction, alpha), *output[1:])
        return _project_out(output, direction, alpha)
    return hook_fn


def build_scaled_ablation_hooks(loaded: LoadedModel, direction: torch.Tensor, alpha: float):
    """Ablate ``alpha``× the projection onto ``direction`` at every residual-stream site.

    Returns ``(fwd_pre_hooks, fwd_hooks)`` in the same shape as
    ``pasta.hooks.build_all_ablation_hooks``, for use with ``pasta.hooks.add_hooks``
    / ``pasta.generation.generate_once``.
    """
    blocks = loaded.block_modules()
    attns = loaded.attn_modules()
    mlps = loaded.mlp_modules()
    pre = [(blocks[i], _input_pre_hook(direction, alpha)) for i in range(loaded.n_layers)]
    post = [(attns[i], _output_hook(direction, alpha)) for i in range(loaded.n_layers)]
    post += [(mlps[i], _output_hook(direction, alpha)) for i in range(loaded.n_layers)]
    return pre, post

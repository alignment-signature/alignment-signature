"""Forward hooks for direction ablation and activation addition.

Ported minimally from the upstream `refusal_direction/pipeline/utils/hook_utils.py`.

`add_hooks` is a context manager that registers forward / forward-pre hooks
and cleans them up on exit. The two builders below construct hooks that:

  - **ablate** a unit-norm direction from the residual stream:
        h <- h - (h · d) d
    Used at every (block-input, attn-output, mlp-output) site to fully remove
    the direction from the model's computation.

  - **activation-add** a vector to the residual stream of a single layer:
        h <- h + coeff * v
    Used as a sanity check (does adding `v_cross` to the base model push it
    toward looking aligned?).
"""

from __future__ import annotations

import contextlib
from typing import Callable

import torch

from pasta.models import LoadedModel


@contextlib.contextmanager
def add_hooks(
    module_forward_pre_hooks: list[tuple[torch.nn.Module, Callable]],
    module_forward_hooks: list[tuple[torch.nn.Module, Callable]],
):
    handles = []
    try:
        for module, hook in module_forward_pre_hooks:
            handles.append(module.register_forward_pre_hook(hook))
        for module, hook in module_forward_hooks:
            handles.append(module.register_forward_hook(hook))
        yield
    finally:
        for h in handles:
            h.remove()


def _project_out(activation: torch.Tensor, direction: torch.Tensor) -> torch.Tensor:
    direction = direction / (direction.norm(dim=-1, keepdim=True) + 1e-8)
    direction = direction.to(activation)
    return activation - (activation @ direction).unsqueeze(-1) * direction


def get_direction_ablation_input_pre_hook(direction: torch.Tensor):
    def hook_fn(module, input):
        if isinstance(input, tuple):
            activation = _project_out(input[0], direction)
            return (activation, *input[1:])
        return _project_out(input, direction)
    return hook_fn


def get_direction_ablation_output_hook(direction: torch.Tensor):
    def hook_fn(module, input, output):
        if isinstance(output, tuple):
            activation = _project_out(output[0], direction)
            return (activation, *output[1:])
        return _project_out(output, direction)
    return hook_fn


def get_activation_addition_input_pre_hook(vector: torch.Tensor, coeff: float):
    def hook_fn(module, input):
        if isinstance(input, tuple):
            activation = input[0]
            activation = activation + coeff * vector.to(activation)
            return (activation, *input[1:])
        return input + coeff * vector.to(input)
    return hook_fn


def build_all_ablation_hooks(loaded: LoadedModel, direction: torch.Tensor):
    """Register direction ablation at every (block input, attn output, mlp output) site."""
    blocks = loaded.block_modules()
    attns = loaded.attn_modules()
    mlps = loaded.mlp_modules()
    pre = [(blocks[i], get_direction_ablation_input_pre_hook(direction)) for i in range(loaded.n_layers)]
    post = [(attns[i], get_direction_ablation_output_hook(direction)) for i in range(loaded.n_layers)]
    post += [(mlps[i], get_direction_ablation_output_hook(direction)) for i in range(loaded.n_layers)]
    return pre, post

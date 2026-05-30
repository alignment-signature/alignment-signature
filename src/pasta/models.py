"""HuggingFace causal-LM loading.

`load_model("meta-llama/...", "cuda:0")` returns a `LoadedModel` carrying the
HF model, tokenizer, device, and convenience accessors for the residual-stream
sites (block / attn / mlp modules) the hook code needs.

Pass `device="auto"` to use `device_map="auto"` for models that don't fit on a
single GPU (e.g. Qwen2.5-72B at bf16). Pair "auto" with `CUDA_VISIBLE_DEVICES`
to control which GPUs the model spans.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


@dataclass
class LoadedModel:
    model: AutoModelForCausalLM
    tokenizer: AutoTokenizer
    device: str
    model_id: str

    @property
    def n_layers(self) -> int:
        return self.model.config.num_hidden_layers

    @property
    def d_model(self) -> int:
        return self.model.config.hidden_size

    def block_modules(self) -> list[torch.nn.Module]:
        return list(self.model.model.layers)

    def attn_modules(self) -> list[torch.nn.Module]:
        return [layer.self_attn for layer in self.model.model.layers]

    def mlp_modules(self) -> list[torch.nn.Module]:
        return [layer.mlp for layer in self.model.model.layers]


def load_model(model_id: str, device: str, dtype=torch.bfloat16) -> LoadedModel:
    tok = AutoTokenizer.from_pretrained(model_id)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    if device == "auto":
        mdl = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=dtype, device_map="auto")
        actual_device = "cuda:0"
    else:
        mdl = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=dtype)
        mdl = mdl.to(device).eval()
        actual_device = device
    mdl.requires_grad_(False)
    return LoadedModel(model=mdl, tokenizer=tok, device=actual_device, model_id=model_id)

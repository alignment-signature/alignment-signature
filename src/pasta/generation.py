"""Ablation-time generation primitives + JSON record builders.

`generate_once` is a thin wrapper around `model.generate` that registers
ablation/steering hooks via `pasta.hooks.add_hooks` for the duration of the
call. It is intended for *post-extraction* generation only — pasta does not
produce the baseline (no-intervention) text that feeds extraction; that
text must be supplied to extraction by an external pipeline.

`build_generation_record` and `build_domain_bundle` produce the JSON shapes
the host repo's Pangram evaluator already understands, so detection scripts
can score ablation outputs directly.
"""

from __future__ import annotations

import torch

from pasta.hooks import add_hooks
from pasta.models import LoadedModel
from pasta.vendor.text_processing import detect_degenerate, truncate_to_complete_sentence


# ---------------------------------------------------------------------------
# Prompt → token ids
# ---------------------------------------------------------------------------
def chat_prompt_ids(loaded: LoadedModel, messages: list[dict]) -> torch.Tensor:
    """Apply the model's chat template and return tokenized prompt ids (1, seq)."""
    ids = loaded.tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True)
    return torch.tensor([ids], device=loaded.device)


def raw_prompt_ids(loaded: LoadedModel, text: str) -> torch.Tensor:
    ids = loaded.tokenizer.encode(text, add_special_tokens=True)
    return torch.tensor([ids], device=loaded.device)


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------
def generate_once(
    loaded: LoadedModel,
    input_ids: torch.Tensor,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    fwd_pre_hooks: list | None = None,
    fwd_hooks: list | None = None,
    seed: int | None = None,
) -> dict:
    """Single-sample generation with optional hooks. Returns dict(raw_text, finish_reason, usage)."""
    fwd_pre_hooks = fwd_pre_hooks or []
    fwd_hooks = fwd_hooks or []
    if seed is not None:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    with add_hooks(fwd_pre_hooks, fwd_hooks):
        out = loaded.model.generate(
            input_ids=input_ids,
            attention_mask=torch.ones_like(input_ids),
            max_new_tokens=max_new_tokens,
            do_sample=temperature > 0.0,
            temperature=max(temperature, 1e-6),
            top_p=top_p,
            pad_token_id=loaded.tokenizer.pad_token_id,
            return_dict_in_generate=True,
        )

    gen_ids = out.sequences[0, input_ids.shape[1]:]
    raw = loaded.tokenizer.decode(gen_ids, skip_special_tokens=True)
    last = gen_ids[-1].item() if gen_ids.numel() > 0 else None
    eos_ids = {loaded.tokenizer.eos_token_id}
    if getattr(loaded.tokenizer, "eot_id", None) is not None:
        eos_ids.add(loaded.tokenizer.eot_id)
    finish = "stop" if last in eos_ids else "length"

    return {
        "raw_text": raw,
        "finish_reason": finish,
        "usage": {
            "prompt_tokens": int(input_ids.shape[1]),
            "completion_tokens": int(gen_ids.shape[0]),
        },
    }


# ---------------------------------------------------------------------------
# Record / bundle builders — schema matches the host repo's Pangram pipeline
# ---------------------------------------------------------------------------
def build_generation_record(
    *,
    strategy_id: str,
    domain: str,
    model_type: str,
    model_id: str,
    sample_id: int,
    original_prefix_id: int,
    prompt_used,
    raw_text: str,
    finish_reason: str,
    usage: dict,
) -> dict:
    processed = truncate_to_complete_sentence(raw_text)
    return {
        "id": f"strategy_{strategy_id}-{domain}-{model_type}-{sample_id}",
        "sample_id": sample_id,
        "original_prefix_id": original_prefix_id,
        "domain": domain,
        "model_type": model_type,
        "model_id": model_id,
        "raw_text": raw_text,
        "processed_text": processed,
        "is_degenerate": detect_degenerate(processed),
        "finish_reason": finish_reason,
        "usage": usage,
        "prompt_used": prompt_used,
    }


def build_domain_bundle(
    *,
    strategy_id: str,
    strategy_name: str,
    domain: str,
    model_type: str,
    model_id: str,
    quantization: str,
    generation_params: dict,
    intervention: dict,
    generations: list[dict],
) -> dict:
    return {
        "domain": domain,
        "model_type": model_type,
        "model_id": model_id,
        "strategy_id": strategy_id,
        "strategy_name": strategy_name,
        "quantization": quantization,
        "generation_params": generation_params,
        "intervention": intervention,
        "num_generations": len(generations),
        "num_degenerate": sum(1 for g in generations if g["is_degenerate"]),
        "generations": generations,
    }

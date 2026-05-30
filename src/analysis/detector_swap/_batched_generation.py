"""Batched chat-template tokenization + generation.

The canonical pasta runner (`pasta.generation.generate_once`) is single-sample.
This analysis runs 7 conditions × N domains × 25 prompts per model and benefits
from batched `model.generate` calls to amortise GPU overhead. The two functions
below mirror `pasta.generation.generate_once` semantics for a batch of prompts.

Local to this analysis only — not promoted to `pasta` until a second caller
needs batched generation.
"""
from __future__ import annotations

import sys
from pathlib import Path

_REPO_SRC = Path(__file__).resolve().parents[3] / "src"
if str(_REPO_SRC) not in sys.path:
    sys.path.insert(0, str(_REPO_SRC))

import torch

from pasta.hooks import add_hooks
from pasta.models import LoadedModel


def chat_prompt_ids_batch(
    loaded: LoadedModel,
    batch_messages: list[list[dict]],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply chat template to each conversation, left-pad to a common length.

    Returns (input_ids, attention_mask) on `loaded.device`. Left-padding is
    required because `tokenizer.padding_side = "left"` is set in load_model().
    """
    seqs: list[list[int]] = []
    for messages in batch_messages:
        seqs.append(loaded.tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True))
    max_len = max(len(s) for s in seqs)
    pad_id = loaded.tokenizer.pad_token_id
    input_ids = torch.full((len(seqs), max_len), pad_id, dtype=torch.long)
    attn = torch.zeros((len(seqs), max_len), dtype=torch.long)
    for i, s in enumerate(seqs):
        input_ids[i, max_len - len(s):] = torch.tensor(s, dtype=torch.long)
        attn[i, max_len - len(s):] = 1
    return input_ids.to(loaded.device), attn.to(loaded.device)


def generate_once_batch(
    loaded: LoadedModel,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    fwd_pre_hooks: list = [],
    fwd_hooks: list = [],
    seed: int | None = None,
) -> list[dict]:
    """Batched generation; returns one dict per row matching `pasta.generation.generate_once`."""
    if seed is not None:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    with add_hooks(fwd_pre_hooks, fwd_hooks):
        out = loaded.model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=temperature > 0.0,
            temperature=max(temperature, 1e-6),
            top_p=top_p,
            pad_token_id=loaded.tokenizer.pad_token_id,
            return_dict_in_generate=True,
        )

    eos_ids = {loaded.tokenizer.eos_token_id}
    if getattr(loaded.tokenizer, "eot_id", None) is not None:
        eos_ids.add(loaded.tokenizer.eot_id)

    results: list[dict] = []
    prompt_len = input_ids.shape[1]
    for row in range(out.sequences.shape[0]):
        gen_ids = out.sequences[row, prompt_len:]
        nonpad_mask = gen_ids != loaded.tokenizer.pad_token_id
        if nonpad_mask.any():
            last_real = int(nonpad_mask.nonzero(as_tuple=True)[0].max().item())
            gen_ids = gen_ids[: last_real + 1]
        raw = loaded.tokenizer.decode(gen_ids, skip_special_tokens=True)
        last = gen_ids[-1].item() if gen_ids.numel() > 0 else None
        finish = "stop" if last in eos_ids else "length"
        row_prompt_len = int(attention_mask[row].sum().item())
        results.append({
            "raw_text": raw,
            "finish_reason": finish,
            "usage": {
                "prompt_tokens": row_prompt_len,
                "completion_tokens": int(gen_ids.shape[0]),
            },
        })
    return results

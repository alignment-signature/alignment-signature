# Measuring, Localizing, and Ablating Alignment Signatures in LLMs

Code and data artifact for the paper *Measuring, Localizing, and Ablating
Alignment Signatures in LLMs* (method: **PASTA**).

## Overview

Post-training alignment leaves a measurable, low-dimensional **signature** in a
model's activations that AI-text detectors rely on. PASTA:

1. **Measures** the signature as a cross-model *alignment direction*
   `v_cross[L] = unit(mean_aligned[L] − mean_base[L])`, estimated from paired
   aligned-model vs. base-model activations.
2. **Localizes** it — finds the layer L\* at which ablating the direction most
   reduces a detector's "AI" score.
3. **Ablates** it at inference — projects the direction out of the residual
   stream (`h ← h − (h·d) d`) and measures the effect on AI-text detection and on
   generation quality.

## Repository layout

```
src/
  pasta/                PASTA library
    extraction.py       estimate v_cross directions (teacher-forced means)
    hooks.py            residual-stream direction-ablation hooks
    generation.py       batched generation under interventions
    models.py           model loading / layer access
    prompts.py          naturalistic generation prompts
    io.py paths.py config.py data.py analysis.py
    vendor/             Pangram detector client + text utilities
  analysis/             one directory per analysis (see below)

data/
  prefixes/             prompt prefixes, 5 domains
  pasta_directions/     best-layer v_cross vector per model family
  generations/          naturalistic generations:
                        aligned / base / pasta_ablated / scaled_ablated
  analysis_results/     per-analysis generations + detection scores
                        (+ direction vectors where a new direction is used)
```

## Analyses

Code in `src/analysis/<name>/`, data in `data/analysis_results/<name>/`.

| analysis | question |
|---|---|
| `detector_quality` | How well does each AI-text detector separate aligned-model text from human / base-model text? |
| `in_domain_pasta` | Per-domain alignment directions, and the best ablation layer for each domain. |
| `per_layer` | Effect of ablating the alignment direction at all layers simultaneously. |
| `scaled_ablation` | Effect of scaling the ablation strength α (under- / full / over-ablation). |
| `random_direction_ablation` | Control: ablating random directions vs. the PASTA direction. |
| `detector_swap` | Different detectors localize the signal to different layers. |
| `generation_quality_analysis` | LLM-judge pairwise fluency / relevance: ablation preserves quality. |

Each analysis's entry-point modules carry a docstring describing how to run them.

## Models, domains, detectors

**Model families (6):** Llama-3.1-8B, OLMo-3-7B, Qwen2.5-7B, Qwen2.5-14B,
Gemma-2-9B, Mistral-7B-v0.3.

**Domains (5):** college essays, creative fiction, news articles, opinion
pieces, scientific abstracts.

**AI-text detectors (6):** Pangram, GPTZero, Originality (closed-source) and
Binoculars, Fast-DetectGPT, ImBD (open-source). Most analyses report a
four-detector panel (Pangram, GPTZero, ImBD, Binoculars); `detector_quality`
reports all six.

## Notes on the data

The shipped data is a **minimal, inspectable sample** — roughly 10 examples per
(model, condition, domain) cell rather than the full evaluation sweeps — intended
for inspection and to keep each analysis self-contained (its generations,
detection scores, and relevant direction vectors).

Reproducing scores for the **closed-source** detectors (Pangram, GPTZero,
Originality) requires your own API keys, which are **not** included; an in-repo
Pangram client is provided under `src/pasta/vendor/`. Running the
generation / ablation pipelines requires a GPU with PyTorch and Transformers.

## Citation

To appear.

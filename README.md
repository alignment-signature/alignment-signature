# Measuring, Localizing, and Ablating Alignment Signatures in LLMs

Code and data artifact for the paper *Measuring, Localizing, and Ablating
Alignment Signatures in LLMs* (method: **PASTA** — Post-training Alignment
Signature Targeted Ablation).

## Overview

Post-training alignment leaves a measurable, low-dimensional **post-training
alignment signature** in a model's activations that AI-text detectors rely on.
PASTA:

1. **Measures** the signature as a cross-model direction — the PASTA direction
   (`v_cross[L]` in the code), `= unit(mean_aligned[L] − mean_base[L])` — estimated
   from paired aligned-model vs. base-model residual activations.
2. **Localizes** it — selects the layer L\* whose ablated direction yields the
   lowest AI-detection rate on a calibration set.
3. **Ablates** it at inference — projects the direction out of the residual
   stream, `h ← h − α (h·d) d` (strength α, default 1), and measures the effect on
   AI-text detection and on generation quality.

The paper reports that this ablation substantially lowers AI-detection rates for
most aligned models, transfers to detectors not used for layer selection, and is
not reproduced by random directions, while largely preserving fluency and
relevance.

## Repository layout

```
src/
  pasta/                PASTA library
    extraction.py       estimate v_cross directions (teacher-forced means)
    hooks.py            residual-stream direction-ablation hooks
    generation.py       single-sample generation under interventions
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
| `per_layer` | Effect of ablating each layer's own direction at all layers at once, vs. the single best-layer PASTA direction. |
| `scaled_ablation` | Effect of scaling the ablation strength α (under- / full / over-ablation). |
| `random_direction_ablation` | Control: ablating random directions vs. the PASTA direction. |
| `detector_swap` | Different detectors localize the signal to different layers. |
| `generation_quality_analysis` | LLM-judge pairwise fluency / relevance: ablation preserves quality. |

Each analysis's entry-point modules carry a docstring describing how to run them.

## Models, domains, detectors

**Models.** Six instruction-tuned models, each paired with its base model, anchor
the PASTA directions and the aligned / base / ablated generation triplets:
Llama-3.1-8B, OLMo-3-7B, Qwen2.5-7B, Qwen2.5-14B, Gemma-2-9B, and Mistral-7B-v0.3.
Individual analyses use subsets of these; `detector_quality` additionally scores
other aligned models (e.g. Tulu-3-8B and smaller Qwen2.5 sizes).

**Domains (5):** college essays, creative fiction, news articles, opinion
pieces, scientific abstracts.

**AI-text detectors (6):** Pangram, GPTZero, Originality (closed-source) and
Binoculars, Fast-DetectGPT, ImBD (open-source). Most analyses report a
four-detector panel (Pangram, GPTZero, ImBD, Binoculars); `detector_quality`
reports all six.

**Scope.** The paper's full evaluation is broader than this artifact — 11 aligned
models, a seventh detector (RAIDAR), and additional analyses (e.g. corpus-affinity
and post-training-stage breakdowns). This repository ships a representative,
inspectable subset.

## Notes on the data

The shipped data is a **minimal, inspectable sample** rather than the full
evaluation sweeps: the per-analysis cells hold on the order of 10 examples each
(the base generation triplets under `data/generations/` keep 25). It is intended
for inspection and to keep each analysis self-contained (its generations,
detection scores, and relevant direction vectors).

Reproducing scores for the **closed-source** detectors (Pangram, GPTZero,
Originality) requires your own API keys, which are **not** included; an in-repo
Pangram client is provided under `src/pasta/vendor/`. Running the
generation / ablation pipelines requires a GPU with PyTorch and Transformers.

## Citation

To appear.

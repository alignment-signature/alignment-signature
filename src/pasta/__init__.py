"""PASTA — Post-training Alignment Steering & Text-detector Ablation.

Library scope:
  1. **Extraction** — given paired (base, aligned) generation records produced
     externally, teacher-force each model on its own (prompt + output) and
     compute the per-layer difference-in-means direction `v_cross`.
  2. **Ablation-time generation** — register residual-stream hooks that project
     `v_cross` out of every block input / attn output / mlp output, then run
     `model.generate(...)` with those hooks installed.

Out of scope: producing the no-intervention (baseline) text that feeds
extraction. That is generic LLM inference and must be supplied by the caller.

Detector: the only AI-text detector currently wired into the pipeline is
**Pangram** (via `src/pasta/vendor/pangram_client.py`, `scripts/evaluate_pangram.py`,
and `pasta.analysis.summarize`). "Best layer" is defined exclusively as the
layer with the lowest mean Pangram `fraction_ai`. Adding a second detector
(Originality, GPTZero, Fast-DetectGPT, etc.) requires a new evaluator script
plus generalizing `pasta.analysis` to read from a configurable detector root.

Public surface:
    from pasta.config import load_config
    from pasta.prompts import load_prefixes, select_prefix_ids, build_strategy_A_prompt
    from pasta.models import LoadedModel, load_model
    from pasta.hooks import build_all_ablation_hooks, get_activation_addition_input_pre_hook
    from pasta.generation import generate_once, chat_prompt_ids, raw_prompt_ids,
                                 build_generation_record, build_domain_bundle
    from pasta.io import write_bundle, write_record_incremental
    from pasta.extraction import (accumulate_means, load_baseline_records,
                                  validate_paired_records, write_directions)
    from pasta.analysis import summarize, render_results_md
    from pasta.data import (prompts_for, prefix_entries, human_continuation_for,
                            load_human, aligned_bundles, base_bundles,
                            pasta_ablated_runs, list_subsets, register_subset)
"""

__version__ = "0.1.0"

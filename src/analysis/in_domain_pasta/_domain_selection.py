r"""Train / test sample selection for the domain-specific direction analysis.

**Test set — Path A canonical (matches every other refactored analysis):**
the first `n_test` ids per domain → ids `[0..n_test-1]`. With `n_test=25`,
test = `[0..24]` per domain. Identical to `scaled_ablation`,
`random_direction_ablation`, `per_layer`, and `per_layer_single_layer_only_ablation`,
so the aligned-baseline + canonical-PASTA-at-L\* anchors on those 25 prompts
can be reused as anchors here without re-scoring.

**Train set — Option B (PASTA's hash-seeded selection on the held-out pool):**
hash-seeded random sample of `n_train` ids drawn from the held-out pool
`[n_test..max_id]` (i.e. ids ≥ `N_TEST_RESERVED`), using the **same algorithm,
same seed, and same domain hash** as `pasta.prompts.select_prefix_ids` — the
function canonical PASTA uses to pick its direction-extraction prompts. The
pool is restricted to ids ≥ `N_TEST_RESERVED` so train ⟂ test by construction.

Picking train this way gives:
  • train ⟂ test (no memorization) — guaranteed by the disjoint pools.
  • Same selection algorithm as canonical PASTA (`select_prefix_ids`,
    `seed=42`, `_stable_hash(domain)` offset) → methodologically consistent
    with the cross-domain `v_cross[L]` runs we compare against.
  • Deterministic + reproducible from one line of code.

With `n_test=25, n_train=25, seed=42` on today's 100-prompt corpus, the
ids are:

  test_ids[domain] = [0..24]  (all 5 domains)
  train_ids[domain] = sorted(rng.sample([25..99], 25))  per-domain hash-seeded

For example, train_ids["college_essays"] starts `[26, 30, 31, 40, 43, ...]`.
"""
from __future__ import annotations

import random

from pasta.prompts import _stable_hash, load_prefixes


N_TEST_RESERVED = 25
TRAIN_SELECTION_SEED = 42


def select_test_prefix_ids(domain: str, n_test: int,
                           prompts_subdir: str | None = None) -> list[int]:
    """First `n_test` ids per domain — the canonical Path A test set."""
    prefixes = load_prefixes(domain, override_subdir=prompts_subdir)
    all_ids = sorted(p["id"] for p in prefixes)
    if n_test > len(all_ids):
        raise SystemExit(f"only {len(all_ids)} ids available, need n_test={n_test}")
    return all_ids[:n_test]


def select_train_prefix_ids(domain: str, n_train: int,
                            prompts_subdir: str | None = None,
                            seed: int = TRAIN_SELECTION_SEED) -> list[int]:
    """Hash-seeded random sample of `n_train` ids from the held-out pool
    `[N_TEST_RESERVED..max_id]`.

    Reproduces `pasta.prompts.select_prefix_ids(domain, n_train, seed)` exactly,
    except the input id pool is the held-out subset (ids ≥ N_TEST_RESERVED)
    rather than the full corpus. Train and test are therefore disjoint by
    construction.
    """
    prefixes = load_prefixes(domain, override_subdir=prompts_subdir)
    all_ids = sorted(p["id"] for p in prefixes)
    pool = [i for i in all_ids if i >= N_TEST_RESERVED]
    if n_train > len(pool):
        raise SystemExit(
            f"need n_train={n_train} train ids per domain but only {len(pool)} "
            f"available in the held-out pool (ids >= {N_TEST_RESERVED})"
        )
    rng = random.Random(seed + _stable_hash(domain))
    return sorted(rng.sample(pool, n_train))

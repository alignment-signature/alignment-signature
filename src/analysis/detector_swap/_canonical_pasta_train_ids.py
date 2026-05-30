"""Canonical PASTA train ids — the **legacy** 5-per-domain extraction prompts.

These are the actual `original_prefix_id`s the canonical PASTA pipeline used
when it produced the per-layer ablation train sweep at
`data/generations/pasta_ablated/<model>/v_cross_L<L>_ablated/<domain>.json`.

They were computed by `select_prefix_ids(domain, n=5, seed=42)` against the
small (~25-30/domain) corpus that existed at extraction time. Today's
100/domain corpus produces *different* ids under the same algorithm — see
`data/SELECTIONS.md:40-77` for the full provenance, and SELECTIONS.md
lines 50-57 for the table:

    | college_essays       | [4, 11, 14, 18, 22]      |
    | news_articles        | [0, 14, 19, 21, 24]      |
    | scientific_abstracts | [11, 13, 20, 22, 26]     |
    | creative_fiction     | [2, 6, 18, 23, 26]       |
    | opinion_pieces       | [5, 15, 17, 26, 29]      |

The detector-swap Step 1 scripts filter the canonical PASTA train sweep
bundles to these ids (a) for the non-best-layer cells (which contain only
these 5/domain samples by construction), the filter is a no-op; (b) for
the best-layer cells (which contain the full 100/domain corpus), the filter
restricts to the 25 prompts canonical PASTA actually trained on.
"""
from __future__ import annotations

CANONICAL_PASTA_TRAIN_IDS: dict[str, list[int]] = {
    "college_essays":       [4, 11, 14, 18, 22],
    "news_articles":        [0, 14, 19, 21, 24],
    "scientific_abstracts": [11, 13, 20, 22, 26],
    "creative_fiction":     [2, 6, 18, 23, 26],
    "opinion_pieces":       [5, 15, 17, 26, 29],
}


def canonical_pasta_train_ids(domain: str) -> set[int]:
    """The 5 canonical-PASTA train ids for `domain` (the legacy ids actually
    on disk in the pasta_ablated train sweep)."""
    if domain not in CANONICAL_PASTA_TRAIN_IDS:
        raise KeyError(f"unknown domain '{domain}'. "
                       f"Known: {list(CANONICAL_PASTA_TRAIN_IDS)}")
    return set(CANONICAL_PASTA_TRAIN_IDS[domain])

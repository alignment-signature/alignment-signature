"""Text post-processing helpers vendored from scripts/helpers.py.

Two functions used by alignment_removal/helpers.py to clean up generated text
before persisting to the bundle JSON:

  - truncate_to_complete_sentence: keep text up to the last sentence boundary
  - detect_degenerate: heuristic for repetition/empty output

The implementations are byte-identical to the upstream copies as of the
vendoring date; see vendor/__init__.py for provenance.
"""

import re


def truncate_to_complete_sentence(text: str) -> str:
    """Truncate text to the last complete sentence."""
    if text.strip().startswith("def ") or text.strip().startswith("class "):
        return text.rstrip()
    matches = list(re.finditer(r'[.!?]["\')\]]?\s', text))
    if matches:
        last_match = matches[-1]
        return text[: last_match.end()].strip()
    return text.strip()


def detect_degenerate(text: str) -> bool:
    """Check if text is degenerate (empty / hard repetition).

    Three-rule heuristic. A text is degenerate iff any rule fires.

    Rule 1 — Empty / trivially short
        `len(text.strip()) < 50` — the model produced essentially nothing.

    Rule 2 — Sentence-level loop (verbatim repeat ≥ 3×)
        After splitting on sentence boundaries (`.!?` followed by whitespace),
        any sentence of length ≥ 40 chars that appears 3 or more times
        verbatim. Tolerates section headers ("1. Introduction") appearing
        once in a TOC and once in the body (count = 2), while flagging true
        sentence-level loops where the same sentence repeats 3+ times.

    Rule 3 — Macro repetition (same 50-char window ≥ 4× anywhere)
        Catches "OPTIONS:" / bulleted-suffix loops and similar where the
        model emits the same 50-char chunk many times. Tolerates topical
        repetition: an academic essay about "Earnings Management" mentioning
        that phrase 9× across 2000 chars is fine — the 50-char windows
        containing it differ from each other because of surrounding context.

    What this rule deliberately does NOT flag (vs. the legacy upstream rule
    which over-flagged topical essays):
    - A noun phrase or proper noun repeating 5+ times across the document
      (legitimate when it's the essay's topic). The upstream rule used
      `text.count(20-char chunk) >= 5` which conflated topical repetition
      with degeneracy; v2's 50-char×4 threshold is conservative enough that
      topical repetition almost never trips it while still catching loops.
    """
    import re as _re
    from collections import Counter as _Counter

    stripped = text.strip()
    if len(stripped) < 50:
        return True

    sents = _re.split(r"(?<=[.!?])\s+", stripped)
    sents = [" ".join(s.split()) for s in sents if len(s.strip()) >= 40]
    if sents and max(_Counter(sents).values()) >= 3:
        return True

    k = 50
    counts: dict[str, int] = {}
    for i in range(0, len(text) - k):
        chunk = text[i:i + k]
        if not chunk.strip():
            continue
        counts[chunk] = counts.get(chunk, 0) + 1
    if counts and max(counts.values()) >= 4:
        return True

    return False

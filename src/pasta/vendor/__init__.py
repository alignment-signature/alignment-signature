"""Vendored copies of upstream code so PASTA is self-contained.

These files were copied from the parent repo at the time of vendoring and are
considered frozen — re-vendor manually if upstream changes. Sources (relative
to the upstream host repo):

  strategy_templates.py — scripts/strategy_templates.py
  text_processing.py    — scripts/helpers.py (truncate_to_complete_sentence, detect_degenerate)
  pangram_client.py     — scripts/detection/evaluate_pangram.py (PangramClient + PANGRAM_API_URL)
"""

"""Common pairwise quality judge — Claude Sonnet 4.6 via the Anthropic Console.

Judges generations **pairwise** on one quality axis defined entirely by the
metric-prompt template you pass in (the repo ships two rubrics: prompt relevance
and fluency). For every input pair the judge is asked twice, swapping the two
texts between the A and B slots (AB and BA order), to cancel position bias.

Inputs
------
--generations  JSONL (one comparison per line) or a JSON list. Each item:
                 text_a, text_b   the two generations to compare        (required)
                 prompt           the user prompt both were written for  (required;
                                  filled into the rubric's {prompt} slot)
                 id               identifier (defaults to the row index)
               Any other fields are carried through verbatim into the output.
--prompt       the metric rubric, e.g. prompts/fluency_pairwise.md or
               prompts/relevance_pairwise.md. Must contain the {prompt},
               {response_a}, {response_b} placeholders and request the
               response_a_score / response_b_score / winner / tie_type schema.
--out          output JSON path.

Auth: ANTHROPIC_API_KEY (Anthropic Console), read from the environment or a
.env at the repo root.

Output (JSON): {"meta": {...}, "results": [ {id, <carried fields>,
  "order_AB": {score_a, score_b, winner, tie_type, reasoning_a, reasoning_b,
               comparative_reasoning} | null,
  "order_BA": { ...same... } | null,
  "errors": [...] }, ... ] }
In order_AB, slot A is text_a and slot B is text_b; in order_BA they are
swapped. Running both orders lets a downstream step cancel A/B position bias.

Usage:
  python judge.py \
    --generations pairs.jsonl \
    --prompt src/analysis/generation_quality_analysis/prompts/relevance_pairwise.md \
    --out relevance_results.json
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import os
import re
import sys
import threading
import time
from pathlib import Path

from tenacity import retry, stop_after_attempt, wait_random_exponential

REPO_ROOT = Path(__file__).resolve().parents[3]

MODEL = "claude-sonnet-4-6"
TEMPERATURE = 0.0
MAX_OUTPUT_TOKENS = 2048
CHAR_CAP = 8000
VALID_SCORES = {1, 2, 3, 4, 5}
SYSTEM_PROMPT = (
    "You are an evaluation judge. Always return exactly one JSON object matching "
    "the schema requested in the prompt and nothing else. Do not include markdown fences."
)
_FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```\s*$", re.IGNORECASE)
PLACEHOLDERS = ("{prompt}", "{response_a}", "{response_b}")


def _load_env_file(path: Path) -> None:
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        k, v = k.strip(), v.strip().strip("'").strip('"')
        if k and k not in os.environ:
            os.environ[k] = v


def _cap(text: str | None, cap: int = CHAR_CAP) -> str:
    if not text:
        return text or ""
    return text if len(text) <= cap else text[:cap] + "\n[…truncated for judge input…]"


def load_generations(path: Path) -> list[dict]:
    """Read comparison items from a .jsonl (one per line) or a .json list."""
    text = path.read_text(encoding="utf-8")
    if path.suffix == ".jsonl":
        items = [json.loads(ln) for ln in text.splitlines() if ln.strip()]
    else:
        data = json.loads(text)
        items = data if isinstance(data, list) else (data.get("examples") or data.get("results") or [])
    out = []
    for i, item in enumerate(items):
        if "text_a" not in item or "text_b" not in item:
            raise SystemExit(f"generations item {i} is missing text_a/text_b")
        out.append({**item, "id": item.get("id", i)})
    return out


def render_prompt(template: str, *, prompt: str, response_a: str, response_b: str) -> str:
    for ph in PLACEHOLDERS:
        if ph not in template:
            raise SystemExit(f"metric prompt is missing the {ph} placeholder")
    return (template.replace("{prompt}", _cap(prompt))
                    .replace("{response_a}", _cap(response_a))
                    .replace("{response_b}", _cap(response_b)))


def parse_judge_json(raw: str) -> dict:
    s = _FENCE_RE.sub("", raw.strip()).strip()
    try:
        return json.loads(s)
    except json.JSONDecodeError:
        pass
    m = re.search(r"\{.*\}", s, flags=re.DOTALL)
    if m:
        try:
            return json.loads(m.group(0))
        except json.JSONDecodeError:
            pass
    raise json.JSONDecodeError(f"unparseable judge output: {s[:200]!r}", s, 0)


def _salvage_score(raw: str, key: str) -> int | None:
    """Recover a 1-5 score from raw text when the JSON is malformed."""
    m = re.search(rf'"{re.escape(key)}"\s*:\s*([1-5])\b', raw)
    return int(m.group(1)) if m else None


_CLIENT_LOCK = threading.Lock()
_CLIENT: dict = {"client": None}


def _client():
    with _CLIENT_LOCK:
        if _CLIENT["client"] is None:
            from anthropic import Anthropic
            key = os.environ.get("ANTHROPIC_API_KEY")
            if not key:
                raise RuntimeError(
                    "ANTHROPIC_API_KEY is not set. Add it to a repo-root .env or "
                    "export it before running.")
            _CLIENT["client"] = Anthropic(api_key=key, max_retries=4)
        return _CLIENT["client"]


@retry(wait=wait_random_exponential(min=1, max=30), stop=stop_after_attempt(5))
def call_judge(prompt: str, *, model: str = MODEL) -> str:
    resp = _client().messages.create(
        model=model,
        max_tokens=MAX_OUTPUT_TOKENS,
        temperature=TEMPERATURE,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": prompt}],
    )
    return "".join(b.text for b in resp.content if getattr(b, "text", None))


def _judge_one(template: str, item: dict, order: str, model: str) -> dict:
    """Run one (item, order) judge call. AB keeps text_a in slot A; BA swaps."""
    text_a, text_b = ((item["text_a"], item["text_b"]) if order == "AB"
                      else (item["text_b"], item["text_a"]))
    meta = {"id": item["id"], "order": order}
    rendered = render_prompt(template, prompt=item.get("prompt", ""),
                             response_a=text_a, response_b=text_b)
    try:
        raw = call_judge(rendered, model=model)
    except Exception as e:
        return {**meta, "ok": False, "error": f"{type(e).__name__}: {str(e)[:300]}", "raw": ""}

    try:
        parsed = parse_judge_json(raw)
    except Exception:
        parsed = {}
    score_a, score_b = parsed.get("response_a_score"), parsed.get("response_b_score")
    if score_a not in VALID_SCORES or score_b not in VALID_SCORES:
        sa, sb = _salvage_score(raw, "response_a_score"), _salvage_score(raw, "response_b_score")
        if sa in VALID_SCORES and sb in VALID_SCORES:
            score_a, score_b = sa, sb
    if score_a not in VALID_SCORES or score_b not in VALID_SCORES:
        return {**meta, "ok": False,
                "error": f"bad scores: a={score_a!r} b={score_b!r}", "raw": raw}

    return {**meta, "ok": True, "raw": raw, "parsed": {
        "score_a": score_a,
        "score_b": score_b,
        "winner": parsed.get("winner"),
        "tie_type": parsed.get("tie_type"),
        "reasoning_a": parsed.get("response_a_assessment"),
        "reasoning_b": parsed.get("response_b_assessment"),
        "comparative_reasoning": parsed.get("comparative_reasoning"),
    }}


_CARRY_SKIP = {"text_a", "text_b", "prompt"}


def run(items: list[dict], template: str, *, model: str, out_path: Path,
        meta: dict, workers: int = 16, checkpoint_every: int = 20) -> tuple[int, int]:
    records = {
        it["id"]: {**{k: v for k, v in it.items() if k not in _CARRY_SKIP},
                   "order_AB": None, "order_BA": None, "errors": []}
        for it in items
    }
    jobs = [(it, order) for it in items for order in ("AB", "BA")]
    lock = threading.Lock()
    done = ok = err = 0
    t0 = time.time()

    def flush() -> None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"meta": {**meta, "completed_calls": done, "elapsed_sec": round(time.time() - t0, 1)},
                   "results": list(records.values())}
        tmp = out_path.with_suffix(out_path.suffix + ".tmp")
        tmp.write_text(json.dumps(payload, indent=2))
        tmp.replace(out_path)

    with cf.ThreadPoolExecutor(max_workers=workers) as pool:
        futs = [pool.submit(_judge_one, template, it, order, model) for it, order in jobs]
        for fut in cf.as_completed(futs):
            r = fut.result()
            with lock:
                rec = records[r["id"]]
                if r["ok"]:
                    rec[f"order_{r['order']}"] = r["parsed"]
                    ok += 1
                else:
                    rec["errors"].append({"order": r["order"], "error": r.get("error"),
                                          "raw": (r.get("raw") or "")[:500]})
                    err += 1
                done += 1
                if done % checkpoint_every == 0:
                    flush()
                if done % 20 == 0 or done == len(jobs):
                    print(f"  [{done}/{len(jobs)}] ok={ok} err={err} "
                          f"elapsed={time.time() - t0:.0f}s", flush=True)
    flush()
    return ok, err


def main() -> None:
    ap = argparse.ArgumentParser(description="Common pairwise quality judge (Claude Sonnet 4.6, Console).")
    ap.add_argument("--generations", type=Path, required=True, help="JSONL/JSON of {text_a, text_b, prompt, ...}.")
    ap.add_argument("--prompt", type=Path, required=True, help="Metric rubric template (fluency_pairwise.md / relevance_pairwise.md).")
    ap.add_argument("--out", type=Path, required=True, help="Output JSON path.")
    ap.add_argument("--model", type=str, default=MODEL)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--checkpoint-every", type=int, default=20)
    ap.add_argument("--n", type=int, default=None, help="Cap to the first N items (pilot).")
    args = ap.parse_args()

    _load_env_file(REPO_ROOT / ".env")
    if not args.prompt.exists():
        raise SystemExit(f"missing metric prompt: {args.prompt}")
    template = args.prompt.read_text(encoding="utf-8")
    for ph in PLACEHOLDERS:
        if ph not in template:
            raise SystemExit(f"{args.prompt} is missing the {ph} placeholder")

    items = load_generations(args.generations)
    if args.n is not None:
        items = items[:args.n]
    if not items:
        raise SystemExit("no generation pairs to judge.")

    meta = {
        "judge_provider": "anthropic_console",
        "judge_model": args.model,
        "judge_temperature": TEMPERATURE,
        "metric_prompt": str(args.prompt),
        "n_items": len(items),
        "n_calls": 2 * len(items),
        "orders": ["AB", "BA"],
    }
    print(f"[judge] {len(items)} pairs x 2 orders = {2 * len(items)} calls "
          f"(model={args.model}, workers={args.workers})", flush=True)
    ok, err = run(items, template, model=args.model, out_path=args.out, meta=meta,
                  workers=args.workers, checkpoint_every=args.checkpoint_every)
    print(f"[judge] wrote {args.out}  (ok={ok} err={err})", flush=True)
    if err:
        sys.exit(2)


if __name__ == "__main__":
    main()

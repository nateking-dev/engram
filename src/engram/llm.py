"""Cached Claude access, used by the summarization baseline (and later the outer tier).

Summarization is the incumbent we are trying to beat, so it must be a *real* compressor,
not a strawman. We instruct Claude to preserve concrete, decision-relevant facts within a
token budget -- i.e. we give the incumbent its best shot. Results are disk-cached keyed by
(model, budget, content hash) so a sweep pays for each compression once.

Offline fallback: deterministic extractive summarization (keep the most recent sentences
that fit the budget). Crude, but it lets the harness run and be tested without network.
"""

from __future__ import annotations

import hashlib
import json

from .config import CACHE_DIR, OFFLINE, SUMMARY_MODEL, count_tokens

_SUMMARY_SYS = (
    "You compress a chunk of an ongoing conversation into a running memory note. "
    "Preserve every concrete, decision-relevant fact: names, dates, deadlines, "
    "quantities, preferences, constraints, decisions, and especially any correction "
    "that supersedes an earlier statement (keep the CURRENT value, not the stale one). "
    "Drop pleasantries and resolved small talk. Be terse and factual. "
    "Stay within roughly {budget} tokens."
)


def _cache_path():
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    return CACHE_DIR / "summaries.jsonl"


def _key(model: str, budget: int, text: str) -> str:
    h = hashlib.sha256(f"{model}|{budget}|{text}".encode()).hexdigest()[:20]
    return h


def _load_cache() -> dict[str, str]:
    path = _cache_path()
    out: dict[str, str] = {}
    if path.exists():
        for line in path.read_text().splitlines():
            if line.strip():
                rec = json.loads(line)
                out[rec["k"]] = rec["v"]
    return out


def _append_cache(key: str, value: str) -> None:
    with _cache_path().open("a") as fh:
        fh.write(json.dumps({"k": key, "v": value}) + "\n")


def _offline_summary(text: str, budget: int) -> str:
    # Keep most-recent sentences that fit -- a recency-biased extractive baseline.
    sentences = [s.strip() for s in text.replace("\n", " ").split(". ") if s.strip()]
    kept: list[str] = []
    used = 0
    for s in reversed(sentences):
        tk = count_tokens(s)
        if used + tk > budget:
            break
        kept.append(s)
        used += tk
    kept.reverse()
    return ". ".join(kept)


def summarize(text: str, budget: int, model: str | None = None) -> str:
    model = model or SUMMARY_MODEL
    key = _key("offline" if OFFLINE else model, budget, text)
    cache = _load_cache()
    if key in cache:
        return cache[key]

    if OFFLINE:
        summary = _offline_summary(text, budget)
    else:
        summary = _claude_summary(text, budget, model)

    _append_cache(key, summary)
    return summary


def _claude_summary(text: str, budget: int, model: str) -> str:
    from anthropic import Anthropic

    client = Anthropic()
    resp = client.messages.create(
        model=model,
        max_tokens=min(4096, int(budget * 1.5) + 64),
        system=_SUMMARY_SYS.format(budget=budget),
        messages=[{"role": "user", "content": f"Conversation chunk to compress:\n\n{text}"}],
    )
    return "".join(b.text for b in resp.content if b.type == "text").strip()

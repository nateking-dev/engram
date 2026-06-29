"""Shared config: paths, tokenizer, environment toggles."""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SCENARIO_DIR = ROOT / "data" / "scenarios"
CACHE_DIR = ROOT / ".cache"
RESULTS_DIR = ROOT / "results"

# When set, all model access (embeddings, summarization) uses deterministic offline
# stand-ins instead of the network. Lets the inner loop + tests run in CI with no keys.
OFFLINE = os.environ.get("ENGRAM_OFFLINE", "") not in ("", "0", "false", "False")

EMBED_MODEL = os.environ.get("ENGRAM_EMBED_MODEL", "text-embedding-3-small")
SUMMARY_MODEL = os.environ.get("ENGRAM_SUMMARY_MODEL", "claude-haiku-4-5-20251001")


@lru_cache(maxsize=1)
def _encoder():
    import tiktoken

    try:
        return tiktoken.get_encoding("cl100k_base")
    except Exception:  # pragma: no cover - network-less environments
        return None


def count_tokens(text: str) -> int:
    enc = _encoder()
    if enc is None:
        # ~4 chars/token is the standard rough approximation.
        return max(1, round(len(text) / 4))
    return len(enc.encode(text))

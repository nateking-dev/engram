"""Cached text embeddings.

Real path: OpenAI text-embedding-3-small. Offline path: a deterministic hashing
bag-of-words vector that still carries lexical-overlap signal -- crude, but enough that
the embedding-only policy and spreading activation produce meaningful (not random)
rankings in CI and tests. Real frontier runs use the real embedder.

All embeddings are disk-cached keyed by (model, text) so a sweep that re-embeds the same
turns thousands of times pays the API cost once.
"""

from __future__ import annotations

import hashlib
import json
import re

import numpy as np

from .config import CACHE_DIR, EMBED_MODEL, OFFLINE

_OFFLINE_DIM = 512
_WORD = re.compile(r"[a-z0-9']+")
_mem_cache: dict[str, np.ndarray] = {}


def _cache_path(model: str):
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    return CACHE_DIR / f"emb_{model.replace('/', '_')}.jsonl"


def _key(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def _offline_embed(text: str) -> np.ndarray:
    vec = np.zeros(_OFFLINE_DIM, dtype=np.float32)
    words = _WORD.findall(text.lower())
    for w in words:
        h = int(hashlib.md5(w.encode()).hexdigest(), 16)
        vec[h % _OFFLINE_DIM] += 1.0
        # a second hashed slot reduces collisions / sharpens overlap signal
        vec[(h // _OFFLINE_DIM) % _OFFLINE_DIM] += 0.5
    n = np.linalg.norm(vec)
    return vec / n if n > 0 else vec


def _load_disk_cache(model: str) -> dict[str, np.ndarray]:
    path = _cache_path(model)
    out: dict[str, np.ndarray] = {}
    if path.exists():
        for line in path.read_text().splitlines():
            if not line.strip():
                continue
            rec = json.loads(line)
            out[rec["k"]] = np.asarray(rec["v"], dtype=np.float32)
    return out


def _append_disk_cache(model: str, key: str, vec: np.ndarray) -> None:
    with _cache_path(model).open("a") as fh:
        fh.write(json.dumps({"k": key, "v": [round(float(x), 6) for x in vec]}) + "\n")


def embed(texts: list[str]) -> np.ndarray:
    """Return an (n, d) array of L2-normalized embeddings, one row per input text."""
    if not texts:
        return np.zeros((0, _OFFLINE_DIM), dtype=np.float32)

    model = "offline" if OFFLINE else EMBED_MODEL
    disk = _load_disk_cache(model)

    out: list[np.ndarray | None] = [None] * len(texts)
    todo: list[tuple[int, str]] = []
    for i, t in enumerate(texts):
        k = _key(t)
        if k in _mem_cache:
            out[i] = _mem_cache[k]
        elif k in disk:
            _mem_cache[k] = disk[k]
            out[i] = disk[k]
        else:
            todo.append((i, t))

    if todo:
        if OFFLINE:
            vecs = [_offline_embed(t) for _, t in todo]
        else:
            vecs = _openai_embed([t for _, t in todo])
        for (i, t), v in zip(todo, vecs):
            v = v.astype(np.float32)
            k = _key(t)
            _mem_cache[k] = v
            _append_disk_cache(model, k, v)
            out[i] = v

    return np.vstack([o for o in out])  # type: ignore[arg-type]


def _openai_embed(texts: list[str]) -> list[np.ndarray]:
    from openai import OpenAI

    client = OpenAI()
    # batch to stay well under request limits
    out: list[np.ndarray] = []
    for i in range(0, len(texts), 256):
        chunk = texts[i : i + 256]
        resp = client.embeddings.create(model=EMBED_MODEL, input=chunk)
        for d in resp.data:
            v = np.asarray(d.embedding, dtype=np.float32)
            v /= np.linalg.norm(v) or 1.0
            out.append(v)
    return out


def cosine(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Cosine similarity of one query vector `a` (d,) against rows of `b` (n, d).
    Inputs are assumed L2-normalized (embed() guarantees this)."""
    if b.shape[0] == 0:
        return np.zeros(0, dtype=np.float32)
    return b @ a

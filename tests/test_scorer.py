"""Scorer + policy behavior. Runs offline (deterministic embeddings/summaries)."""

import os

os.environ.setdefault("ENGRAM_OFFLINE", "1")

from engram.policies import (  # noqa: E402
    EmbeddingRetrieval,
    Engram,
    FullContext,
    SalienceOnly,
    SlidingWindow,
)
from engram.scenarios import load_scenarios  # noqa: E402
from engram.scorer import aggregate, evaluate, score_probe  # noqa: E402


def _scen(sid="medical-intake"):
    return next(s for s in load_scenarios() if s.id == sid)


def test_full_context_recalls_everything_at_max_cost():
    s = _scen()
    a = aggregate(evaluate([FullContext()], [s]), "full-context")
    assert a.recall == 1.0  # recall ceiling
    # full context contains the stale value too -> supersession pollution is maximal
    assert a.stale_rate == 1.0


def test_sliding_window_misses_distant_plant():
    s = _scen()
    # stated-once is planted ~30 turns before the probe; a tiny window can't see it.
    r = score_probe(SlidingWindow(3), s, s.probes[0])
    assert r.recall == 0
    assert r.window_size <= 3


def test_window_cost_increases_with_k():
    s = _scen()
    small = score_probe(EmbeddingRetrieval(2), s, s.probes[0]).window_cost
    big = score_probe(EmbeddingRetrieval(10), s, s.probes[0]).window_cost
    assert big > small


def test_window_contents_logged_for_attribution():
    s = _scen()
    r = score_probe(EmbeddingRetrieval(5), s, s.probes[0])
    assert isinstance(r.window_source_ids, list)
    assert len(r.window_source_ids) == r.window_size
    assert r.matched_by in {"exact", "summary", "none"}


def test_decay_destroys_stated_once_recall():
    # The headline invariant: with everything else fixed, cranking decay should not
    # IMPROVE stated-once recall, and at high decay it should collapse it.
    scens = load_scenarios()
    lo = aggregate(evaluate([Engram(k=8, decay=0.0, label="d0")], scens), "d0")
    hi = aggregate(evaluate([Engram(k=8, decay=1.5, label="d15")], scens), "d15")
    assert lo.recall_stated_once >= hi.recall_stated_once
    assert hi.recall_stated_once < lo.recall_stated_once  # strict collapse


def test_salience_only_does_not_ace():
    # Anti-shiny-needle invariant baked into CI: the query-blind salience baseline must
    # stay well below the full-context ceiling, or the eval measures nothing.
    scens = load_scenarios()
    best = max(
        aggregate(evaluate([SalienceOnly(k)], scens), f"salience-only-{k}").recall
        for k in (3, 5, 8)
    )
    assert best < 0.6

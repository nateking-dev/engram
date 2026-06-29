"""Retention policies under test.

Every policy answers the same question: at decision turn T, what goes in the context
window? It returns a `Window` -- a list of `Chunk`s with a token cost. The scorer then
asks whether the probe's target fact is recoverable from that window, and what fraction of
the window is junk. Billing every policy through the same Window makes the recall-vs-cost
frontier an apples-to-apples comparison.

Two families:
  * selection policies (full / sliding / embedding / engram) put *original turns* in the
    window -- recall is exact (target turn id present or not).
  * the summarization policy puts a *rewritten summary* in the window -- recall is measured
    by whether the fact survived compression (the scorer handles this).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..activation import Candidate, activation, heuristic_importance
from ..embeddings import embed
from ..llm import summarize
from ..types import Scenario


@dataclass
class Chunk:
    source_id: str       # original turn id, or "summary"
    text: str
    token_count: int
    is_summary: bool = False


@dataclass
class Window:
    chunks: list[Chunk] = field(default_factory=list)

    @property
    def cost(self) -> int:
        return sum(c.token_count for c in self.chunks)

    @property
    def source_ids(self) -> set[str]:
        return {c.source_id for c in self.chunks if not c.is_summary}


class Policy:
    name: str = "policy"

    def window(self, scenario: Scenario, decision_turn_id: str, query: str) -> Window:
        raise NotImplementedError


# --- baselines ------------------------------------------------------------------------

class FullContext(Policy):
    """Recall ceiling and cost ceiling. The eval only matters where this won't fit."""
    name = "full-context"

    def window(self, scenario, decision_turn_id, query):
        hist = scenario.turns_before(decision_turn_id)
        return Window([Chunk(t.id, t.text, t.token_count) for t in hist])


class SlidingWindow(Policy):
    """The dumb thing decay must beat: keep the last N turns."""

    def __init__(self, n: int):
        self.n = n
        self.name = f"sliding-{n}"

    def window(self, scenario, decision_turn_id, query):
        hist = scenario.turns_before(decision_turn_id)[-self.n :]
        return Window([Chunk(t.id, t.text, t.token_count) for t in hist])


class EmbeddingRetrieval(Policy):
    """Plain semantic search: top-k turns by cosine to the query. No decay, no graph.
    Isolates whether decay + importance add anything over vanilla RAG."""

    def __init__(self, k: int):
        self.k = k
        self.name = f"embedding-{k}"

    def window(self, scenario, decision_turn_id, query):
        hist = scenario.turns_before(decision_turn_id)
        if not hist:
            return Window([])
        qv = embed([query])[0]
        mat = embed([t.text for t in hist])
        sims = mat @ qv
        top = np.argsort(-sims)[: self.k]
        return Window([Chunk(hist[i].id, hist[i].text, hist[i].token_count) for i in top])


class Summarization(Policy):
    """The incumbent: keep a recent verbatim buffer, compress everything older into a
    running summary of fixed budget. The baseline that can kill the project."""

    def __init__(self, buffer_turns: int, summary_budget: int):
        self.buffer_turns = buffer_turns
        self.summary_budget = summary_budget
        self.name = f"summarize-b{buffer_turns}-s{summary_budget}"

    def window(self, scenario, decision_turn_id, query):
        hist = scenario.turns_before(decision_turn_id)
        if not hist:
            return Window([])
        buffer = hist[-self.buffer_turns :] if self.buffer_turns else []
        older = hist[: len(hist) - len(buffer)]
        chunks = []
        if older:
            text = "\n".join(f"[{t.role}] {t.text}" for t in older)
            summary = summarize(text, self.summary_budget)
            if summary:
                from ..config import count_tokens

                chunks.append(Chunk("summary", summary, count_tokens(summary), is_summary=True))
        chunks += [Chunk(t.id, t.text, t.token_count) for t in buffer]
        return Window(chunks)


class SalienceOnly(Policy):
    """Sanity baseline: rank turns purely by content salience (the importance heuristic),
    ignoring the query and recency. If THIS aces recall, the planted needles are shiny --
    trivially findable by specificity alone -- and the eval measures nothing. It must lose."""

    def __init__(self, k: int):
        self.k = k
        self.name = f"salience-only-{k}"

    def window(self, scenario, decision_turn_id, query):
        hist = scenario.turns_before(decision_turn_id)
        if not hist:
            return Window([])
        scored = sorted(hist, key=lambda t: heuristic_importance(t.text), reverse=True)
        top = scored[: self.k]
        return Window([Chunk(t.id, t.text, t.token_count) for t in top])


# --- the proposed system --------------------------------------------------------------

class Engram(Policy):
    """ACT-R activation retrieval: base-level decay + importance + spreading, top-k."""

    def __init__(self, k: int, decay: float = 0.5, beta: float = 1.0, gamma: float = 1.0,
                 oracle_importance: dict[str, float] | None = None, label: str | None = None):
        self.k = k
        self.decay = decay
        self.beta = beta
        self.gamma = gamma
        self.oracle = oracle_importance
        self.name = label or f"engram-k{k}-d{decay}"

    def window(self, scenario, decision_turn_id, query):
        hist = scenario.turns_before(decision_turn_id)
        if not hist:
            return Window([])
        decision_index = scenario.turn_index(decision_turn_id)
        mat = embed([t.text for t in hist])
        qv = embed([query])[0]
        cands = []
        for t, e in zip(hist, mat):
            imp = (self.oracle or {}).get(t.id)
            if imp is None:
                imp = heuristic_importance(t.text)
            cands.append(Candidate(t.id, t.index, imp, e))
        scores = activation(cands, decision_index, qv,
                            decay=self.decay, beta=self.beta, gamma=self.gamma)
        top = np.argsort(-scores)[: self.k]
        return Window([Chunk(hist[i].id, hist[i].text, hist[i].token_count) for i in top])

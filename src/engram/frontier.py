"""Sweep retention knobs and produce the recall-vs-window-cost frontier.

The deliverable is NOT a single number. It's a frontier: one curve per policy family,
each point an operating choice (how big a window). Collapsing it to a scalar would hide the
precision/recall tradeoff that *is* the question. We also run ablations of the proposed
system (-decay / -importance / -spreading) as their own curves, so the table can tell us
which knob is load-bearing -- and can embarrass the hypothesis if a knob does nothing.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

from .policies import (
    EmbeddingRetrieval,
    Engram,
    FullContext,
    Policy,
    SlidingWindow,
    Summarization,
)
from .scenarios import load_scenarios
from .scorer import PolicyPoint, ProbeResult, aggregate, evaluate
from .types import Scenario


@dataclass
class Curve:
    family: str
    points: list[PolicyPoint]

    def as_dict(self) -> dict:
        return {"family": self.family, "points": [asdict(p) for p in self.points]}


def build_sweep(oracle_importance: dict[str, float] | None = None) -> dict[str, list[Policy]]:
    """Knob grids per policy family. Each list sweeps the window-size knob to trace a curve
    across the cost axis."""
    K = [1, 3, 5, 8, 12]
    return {
        "full-context": [FullContext()],
        "sliding-window": [SlidingWindow(n) for n in [3, 5, 8, 12, 20]],
        "embedding": [EmbeddingRetrieval(k) for k in K],
        "summarization": [
            Summarization(buffer_turns=b, summary_budget=s)
            for (b, s) in [(2, 60), (3, 120), (4, 200), (6, 320), (8, 480)]
        ],
        "engram": [Engram(k=k, decay=0.5, beta=1.0, gamma=1.0) for k in K],
        # --- ablations: knock out one component each, same k grid ---
        "engram(-decay)": [
            Engram(k=k, decay=0.0, beta=1.0, gamma=1.0, label=f"engram(-decay)-k{k}") for k in K
        ],
        "engram(-importance)": [
            Engram(k=k, decay=0.5, beta=0.0, gamma=1.0, label=f"engram(-importance)-k{k}")
            for k in K
        ],
        "engram(-spreading)": [
            Engram(k=k, decay=0.5, beta=1.0, gamma=0.0, label=f"engram(-spreading)-k{k}")
            for k in K
        ],
    }


def run_frontier(
    scenarios: list[Scenario] | None = None,
    sweep: dict[str, list[Policy]] | None = None,
) -> tuple[list[Curve], list[ProbeResult]]:
    scenarios = scenarios or load_scenarios()
    sweep = sweep or build_sweep()
    all_policies = [p for ps in sweep.values() for p in ps]
    results = evaluate(all_policies, scenarios)
    curves = []
    for family, pols in sweep.items():
        pts = sorted((aggregate(results, p.name) for p in pols), key=lambda x: x.mean_cost)
        curves.append(Curve(family, pts))
    return curves, results


# --- decay sweep: the "stated-once / supersession vs decay rate" story ----------------

def run_decay_sweep(
    scenarios: list[Scenario] | None = None,
    k: int = 8,
    decays: list[float] | None = None,
) -> list[PolicyPoint]:
    """At fixed k, sweep the decay rate. This is the curve that should show decay
    destroying stated-once recall while leaving recency-relevant facts intact."""
    scenarios = scenarios or load_scenarios()
    decays = decays or [0.0, 0.25, 0.5, 0.75, 1.0, 1.5]
    pols = [Engram(k=k, decay=d, label=f"engram-d{d}") for d in decays]
    results = evaluate(pols, scenarios)
    return [aggregate(results, p.name) for p in pols]

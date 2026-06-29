"""Inner-loop scorer: did the right memory get surfaced into the window?

No generation. For each (scenario, probe, policy) we build the window and ask:
  * recall  -- is the probe's CURRENT answer fact recoverable from the window?
  * stale_present -- did a superseded value leak into the window? (supersession pollution)
  * cost    -- tokens billed for the window (the frontier's x-axis)

Recall is fact-level, not turn-level, so selection policies and the lossy summarizer are
comparable:
  * selection policies: exact -- the target turn id is in the window, or it isn't.
  * summarization: the fact survived compression -- a summary sentence embeds close to the
    target turn's text (threshold TAU).

Every window's source ids and the match verdict are logged on the result, so a later pass
can split not-retrieved (ranker fault) from retrieved-but-ignored (generation fault).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

import numpy as np

from .config import OFFLINE
from .embeddings import embed
from .policies import Policy, Window
from .types import Importance, Probe, ProbeType, Scenario

# Survived-compression threshold for summary recall. OpenAI 3-small paraphrase cosine runs
# ~0.5-0.8; unrelated ~0.1-0.3. Offline hashing embeddings need a lower bar. Tunable.
TAU = 0.45 if OFFLINE else 0.55
_SENT = re.compile(r"(?<=[.!?])\s+")


@dataclass
class ProbeResult:
    scenario_id: str
    probe_id: str
    probe_type: str
    importance: str
    policy: str
    distance: int           # probe_turn index - plant index (recall-vs-distance is the science)
    n_distractors: int      # superseded/contradictory values planted before the probe
    recall: int             # 1 if current fact recoverable from window
    stale_present: int      # 1 if a superseded value leaked into the window
    window_cost: int        # tokens billed
    window_size: int        # number of chunks
    matched_by: str         # "exact" | "summary" | "none" -- attribution
    window_source_ids: list[str] = field(default_factory=list)


def _max_sentence_sim(summary_text: str, target_text: str) -> float:
    sents = [s for s in _SENT.split(summary_text) if s.strip()]
    if not sents:
        return 0.0
    tv = embed([target_text])[0]
    mat = embed(sents)
    return float(np.max(mat @ tv))


def _recoverable(window: Window, scenario: Scenario, turn_ids: list[str]) -> tuple[bool, str]:
    """Is any of `turn_ids` (the current answer-bearing turns) recoverable from the window?
    Returns (hit, how)."""
    present = window.source_ids
    if any(t in present for t in turn_ids):
        return True, "exact"
    # fall back to survived-compression check against summary chunks
    summaries = [c for c in window.chunks if c.is_summary]
    if summaries:
        for tid in turn_ids:
            target_text = next(t.text for t in scenario.turns if t.id == tid)
            for c in summaries:
                if _max_sentence_sim(c.text, target_text) >= TAU:
                    return True, "summary"
    return False, "none"


def score_probe(policy: Policy, scenario: Scenario, probe: Probe) -> ProbeResult:
    window = policy.window(scenario, probe.probe_turn, probe.question)
    recall, how = _recoverable(window, scenario, probe.target_turns)
    stale, _ = (False, "") if not probe.distractor_turns else \
        _recoverable(window, scenario, probe.distractor_turns)

    plant_idx = scenario.turn_index(probe.plant_turn or probe.target_turns[0])
    probe_idx = scenario.turn_index(probe.probe_turn)
    return ProbeResult(
        scenario_id=scenario.id,
        probe_id=probe.id,
        probe_type=probe.type.value,
        importance=probe.importance.value,
        policy=policy.name,
        distance=probe_idx - plant_idx,
        n_distractors=len(probe.distractor_turns),
        recall=int(recall),
        stale_present=int(bool(stale)),
        window_cost=window.cost,
        window_size=len(window.chunks),
        matched_by=how,
        window_source_ids=sorted(window.source_ids),
    )


def evaluate(policies: list[Policy], scenarios: list[Scenario]) -> list[ProbeResult]:
    results: list[ProbeResult] = []
    for scen in scenarios:
        for probe in scen.probes:
            for pol in policies:
                results.append(score_probe(pol, scen, probe))
    return results


# --- aggregation ----------------------------------------------------------------------

@dataclass
class PolicyPoint:
    policy: str
    recall: float                 # overall recall@k
    recall_stated_once: float
    recall_supersession: float
    recall_high_importance: float
    stale_rate: float             # supersession pollution
    mean_cost: float              # mean window tokens per decision -- frontier x-axis

    def as_row(self) -> dict:
        return self.__dict__.copy()


def _mean(xs: list[float]) -> float:
    return float(np.mean(xs)) if xs else 0.0


def aggregate(results: list[ProbeResult], policy: str) -> PolicyPoint:
    rs = [r for r in results if r.policy == policy]
    return PolicyPoint(
        policy=policy,
        recall=_mean([r.recall for r in rs]),
        recall_stated_once=_mean([r.recall for r in rs if r.probe_type == ProbeType.STATED_ONCE]),
        recall_supersession=_mean([r.recall for r in rs if r.probe_type == ProbeType.SUPERSESSION]),
        recall_high_importance=_mean([r.recall for r in rs if r.importance == Importance.HIGH]),
        stale_rate=_mean([r.stale_present for r in rs
                          if r.probe_type == ProbeType.SUPERSESSION]),
        mean_cost=_mean([r.window_cost for r in rs]),
    )

"""Build the three window conditions for the supersession kill-test.

The Tier-1 study (see ../WHITEPAPER.md §4.7) found embedding retrieval recalls the
*current* value with recall ~1.0 but also drags the *stale* value into the window 73% of
the time -- its windows usually contain BOTH the superseded value X and its correction Y.
The open question (§6, last bullet; §7.6) is whether that stale presence actually changes
what the model *answers*. This module constructs the windows that test it.

For each supersession probe we take embedding's real retrieved neighborhood at the
decision turn, strip out the two answer-bearing turns (the stale value X and the current
value Y), and rebuild three windows that differ ONLY in which of {X, Y} is present:

    live_only   = neutral neighbors + Y          (control: ceiling / can the model read?)
    stale_only  = neutral neighbors + X          (the live fact was evicted -- silent-stale)
    both        = neutral neighbors + X + Y       (the realistic 0.73 case -- the headline)

Holding the neutral neighbors fixed makes this a clean causal manipulation of stale
presence rather than a confound with window size or content. Every window is presented in
chronological (conversation) order -- the arrangement most favorable to the model, since
the correction Y then appears *after* the stale value X. If stale presence drags the model
even here, the effect is real; if it doesn't, the precision pivot has to answer for it.
"""

from __future__ import annotations

from dataclasses import dataclass

from engram.policies import EmbeddingRetrieval
from engram.types import Probe, Scenario, Turn

CONDITIONS = ("live_only", "stale_only", "both")


@dataclass
class Condition:
    name: str
    turns: list[Turn]          # chronologically ordered window contents
    stale_present: bool
    live_present: bool

    @property
    def cost(self) -> int:
        return sum(t.token_count for t in self.turns)


def _turns_by_ids(scenario: Scenario, ids: list[str]) -> list[Turn]:
    by_id = {t.id: t for t in scenario.turns}
    return [by_id[i] for i in ids if i in by_id]


def neutral_neighbors(scenario: Scenario, probe: Probe, k: int) -> list[Turn]:
    """Embedding's real top-k retrieval at the decision turn, minus the answer-bearing
    turns. These are the held-fixed distractor neighbors shared by live_only and both --
    so the only thing that varies across those two is the presence of Y."""
    answer_ids = set(probe.target_turns) | set(probe.distractor_turns)
    window = EmbeddingRetrieval(k).window(scenario, probe.probe_turn, probe.question)
    return [t for t in _turns_by_ids(scenario, sorted(window.source_ids))
            if t.id not in answer_ids]


def _pre_correction(scenario: Scenario, probe: Probe, neighbors: list[Turn]) -> list[Turn]:
    """Neighbors strictly before the correction. The silent-stale window must contain no
    trace of the current value or the supersession event, otherwise the model can recover
    the current value (it explicitly says "updated to Y") and the axis measures nothing.
    Anything at or after the earliest target (correction) turn is dropped."""
    correction_idx = min(scenario.turn_index(t) for t in probe.target_turns)
    return [t for t in neighbors if t.index < correction_idx]


def _chrono(turns: list[Turn]) -> list[Turn]:
    return sorted(turns, key=lambda t: t.index)


def build_conditions(scenario: Scenario, probe: Probe, k: int = 8) -> dict[str, Condition]:
    """Return {condition_name: Condition} for one supersession probe."""
    neutral = neutral_neighbors(scenario, probe, k)
    stale = _turns_by_ids(scenario, probe.distractor_turns)   # X (superseded)
    live = _turns_by_ids(scenario, probe.target_turns)        # Y (current)

    # live_only / both share the held-fixed neighborhood (vary only Y). stale_only is the
    # silent-stale slice: pre-correction neighbors only, so no later turn re-states Y.
    stale_neutral = _pre_correction(scenario, probe, neutral)

    return {
        "live_only": Condition("live_only", _chrono(neutral + live),
                               stale_present=False, live_present=True),
        "stale_only": Condition("stale_only", _chrono(stale_neutral + stale),
                                stale_present=True, live_present=False),
        "both": Condition("both", _chrono(neutral + stale + live),
                          stale_present=True, live_present=True),
    }


def render_window(turns: list[Turn]) -> str:
    """Format a window as a retrieved-context transcript, the way a memory/RAG system would
    inject it. Roles are kept so the model sees who said what."""
    return "\n".join(f"[{t.role}] {t.text}" for t in turns)

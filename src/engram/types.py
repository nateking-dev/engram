"""Core data model for the Engram retrieval eval.

A *scenario* is a multi-turn conversation. Each turn is the atomic memory unit the
rankers operate over (a conversational chunk). A *probe* is a planted needle with
ground-truth labels: where it was stated, where it was reinforced, where it gets
superseded, and the decision turn at which we test whether the right memory was
surfaced into the window.

This schema is the substance of the eval. NIAH/RULER plant one bland needle; here
every probe is designed to kill a specific retention policy (see ProbeType).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class ProbeType(str, Enum):
    # Stated once at turn ~2, probed at ~50, zero reinforcement. Frequency=1,
    # importance=high. The case decay destroys -- the headline.
    STATED_ONCE = "stated_once"
    # Value X at turn a, corrected to Y at turn b>a, probed at c>b. Tests whether we
    # retrieve the *current* value, not the most-reinforced one. Frequency-based decay
    # actively fails this (the stale value got more mentions).
    SUPERSESSION = "supersession"
    # Semantically adjacent fact with opposite implication ("allergic to penicillin"
    # then "tolerated amoxicillin"). Tests whether spreading pulls the wrong neighbor.
    INTERFERENCE = "interference"
    # A resolved tangent that is now genuinely dead. Punishes over-retention so we are
    # actually measuring precision, not just recall.
    SHOULD_FORGET = "should_forget"
    # Live within task A, dead once A closes. The only probe that discriminates
    # time-conditioned from task-conditioned decay. (Layer two -- defined, not yet used.)
    TASK_BOUNDARY = "task_boundary"


class Importance(str, Enum):
    HIGH = "high"
    LOW = "low"


class Status(str, Enum):
    LIVE = "live"   # the fact is current and should be retrievable
    STALE = "stale"  # superseded; retrieving it as the answer is wrong
    DEAD = "dead"   # resolved/irrelevant; retaining it is a precision cost


@dataclass
class Turn:
    """One conversational chunk -- the atomic unit the rankers score and the window
    is billed for."""
    id: str
    role: str          # "user" | "assistant"
    text: str
    index: int = -1    # position in the conversation, filled in by the loader
    token_count: int = 0  # filled in by the loader via the tokenizer


@dataclass
class Probe:
    """A planted needle with ground-truth relevance labels."""
    id: str
    type: ProbeType
    importance: Importance
    question: str              # the retrieval cue posed at the decision turn
    expected_answer: str       # gold answer (used by the outer/end-to-end tier)
    probe_turn: str            # decision turn T at which retrieval is tested
    target_turns: list[str]    # turn id(s) carrying the *current* answer-bearing fact
    plant_turn: str = ""       # where the fact is first stated (defaults to target)
    reference_turns: list[str] = field(default_factory=list)  # reinforcement turns
    distractor_turns: list[str] = field(default_factory=list)  # stale/contradictory values
    expected_status: Status = Status.LIVE
    superseded_by: str | None = None  # id of the probe that supersedes this one
    task_id: str | None = None

    def presentation_turns(self) -> list[str]:
        """All turns at which this fact was presented (for ACT-R base-level activation):
        the plant plus every reinforcement."""
        plant = self.plant_turn or (self.target_turns[0] if self.target_turns else "")
        seen, out = set(), []
        for t in [plant, *self.reference_turns]:
            if t and t not in seen:
                seen.add(t)
                out.append(t)
        return out


@dataclass
class Scenario:
    id: str
    title: str
    turns: list[Turn]
    probes: list[Probe]

    def turn_index(self, turn_id: str) -> int:
        for t in self.turns:
            if t.id == turn_id:
                return t.index
        raise KeyError(f"{self.id}: no turn {turn_id!r}")

    def turns_before(self, turn_id: str, inclusive: bool = False) -> list[Turn]:
        """Turns visible at decision turn `turn_id` -- i.e. the conversation history a
        policy is allowed to draw its window from."""
        cut = self.turn_index(turn_id)
        return [t for t in self.turns if (t.index <= cut if inclusive else t.index < cut)]

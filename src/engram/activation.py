"""ACT-R declarative-memory activation.

Activation of a memory chunk i at decision turn T:

    A_i = B_i  +  beta * importance_i  +  gamma * sim(query_T, i)
          \___/    \________________/      \________________/
        base-level    importance offset       spreading activation
        (recency/      (the hypothesized       (semantic relevance of
         decay)         missing axis)           the cue to the chunk)

Base-level (single-presentation form, since each turn is its own chunk):
    B_i = -d * ln(max(1, T - t_i))
with decay d. d=0 removes decay entirely (the -decay ablation). Larger d punishes old
chunks harder -- this is the knob the "stated-once-matters-forever" probe is designed to
break.

Honesty constraint: every input here is derivable from the conversation alone. importance
comes from a content heuristic, NOT from the probe's ground-truth label. An oracle override
exists only for the idealized "if we knew importance perfectly" ablation, and is opt-in.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np


@dataclass
class Candidate:
    turn_id: str
    turn_index: int
    importance: float            # in [0, 1], heuristic (or oracle) salience
    embedding: np.ndarray        # L2-normalized


# --- importance heuristic -------------------------------------------------------------
# Real systems must *infer* what matters; they can't read the eval's labels. These signals
# are deliberately simple and deployable: explicit salience markers, commitments, and
# concrete quantities/dates are the things a human would jot down.
_REMEMBER = re.compile(r"\b(remember|important|note that|don't forget|make sure|always|never|"
                       r"critical|key thing|for the record|heads up)\b", re.I)
_COMMIT = re.compile(r"\b(must|need to|deadline|due|by (mon|tue|wed|thu|fri|sat|sun|next|the)|"
                     r"allerg|prefer|require|can't|cannot|only|budget)\b", re.I)
_NUM = re.compile(r"\b(\d[\d,.:]*|\$\d|monday|tuesday|wednesday|thursday|friday|saturday|"
                  r"sunday|january|february|march|april|may|june|july|august|september|"
                  r"october|november|december)\b", re.I)


def heuristic_importance(text: str) -> float:
    score = 0.0
    if _REMEMBER.search(text):
        score += 0.6
    if _COMMIT.search(text):
        score += 0.3
    if _NUM.search(text):
        score += 0.2
    return min(1.0, score)


def base_level(turn_index: int, decision_index: int, decay: float) -> float:
    age = max(1, decision_index - turn_index)
    if decay == 0.0:
        return 0.0
    return -decay * np.log(age)


def activation(
    candidates: list[Candidate],
    decision_index: int,
    query_vec: np.ndarray,
    *,
    decay: float = 0.5,
    beta: float = 1.0,
    gamma: float = 1.0,
) -> np.ndarray:
    """Return an activation score per candidate (parallel to `candidates`)."""
    if not candidates:
        return np.zeros(0, dtype=np.float32)
    embs = np.vstack([c.embedding for c in candidates])
    sims = embs @ query_vec  # cosine; inputs normalized
    scores = np.empty(len(candidates), dtype=np.float32)
    for i, c in enumerate(candidates):
        b = base_level(c.turn_index, decision_index, decay)
        scores[i] = b + beta * c.importance + gamma * float(sims[i])
    return scores

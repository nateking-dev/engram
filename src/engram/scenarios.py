"""Load and validate hand-authored scenarios from YAML.

Validation is strict on purpose: a scenario with a probe pointing at a turn that does
not exist, or a supersession probe with no distractor (stale value) to retrieve over,
is a silent corruption of the signal. We fail loudly at load time instead.
"""

from __future__ import annotations

from pathlib import Path

import yaml

from .config import SCENARIO_DIR, count_tokens
from .types import Importance, Probe, ProbeType, Scenario, Status, Turn


class ScenarioError(ValueError):
    pass


def _build_turns(raw_turns: list[dict], sid: str) -> list[Turn]:
    turns: list[Turn] = []
    seen: set[str] = set()
    for i, rt in enumerate(raw_turns):
        tid = str(rt["id"])
        if tid in seen:
            raise ScenarioError(f"{sid}: duplicate turn id {tid!r}")
        seen.add(tid)
        text = rt["text"]
        turns.append(
            Turn(
                id=tid,
                role=rt.get("role", "user"),
                text=text,
                index=i,
                token_count=count_tokens(text),
            )
        )
    return turns


def _build_probe(rp: dict, sid: str, turn_ids: set[str]) -> Probe:
    pid = str(rp["id"])
    ptype = ProbeType(rp["type"])
    target_turns = [str(t) for t in rp["target_turns"]]
    plant_turn = str(rp.get("plant_turn", target_turns[0]))
    reference_turns = [str(t) for t in rp.get("reference_turns", [])]
    distractor_turns = [str(t) for t in rp.get("distractor_turns", [])]

    referenced = {plant_turn, *target_turns, *reference_turns, *distractor_turns,
                  str(rp["probe_turn"])}
    missing = referenced - turn_ids
    if missing:
        raise ScenarioError(f"{sid}/{pid}: references unknown turns {sorted(missing)}")

    if ptype is ProbeType.SUPERSESSION and not distractor_turns:
        raise ScenarioError(
            f"{sid}/{pid}: supersession probe must list distractor_turns (the stale "
            f"value), otherwise there is nothing to retrieve *over*."
        )

    probe = Probe(
        id=pid,
        type=ptype,
        importance=Importance(rp.get("importance", "high")),
        question=rp["question"],
        expected_answer=rp["expected_answer"],
        probe_turn=str(rp["probe_turn"]),
        target_turns=target_turns,
        plant_turn=plant_turn,
        reference_turns=reference_turns,
        distractor_turns=distractor_turns,
        expected_status=Status(rp.get("expected_status", "live")),
        superseded_by=rp.get("superseded_by"),
        task_id=rp.get("task_id"),
    )
    return probe


def load_scenario_file(path: Path) -> Scenario:
    raw = yaml.safe_load(path.read_text())
    sid = str(raw["id"])
    turns = _build_turns(raw["turns"], sid)
    turn_ids = {t.id for t in turns}
    probes = [_build_probe(rp, sid, turn_ids) for rp in raw["probes"]]
    if not probes:
        raise ScenarioError(f"{sid}: scenario has no probes")

    # The probe_turn must come strictly after every presentation/distractor, else we'd
    # be "testing" a fact the policy can already see in the most recent turn.
    scen = Scenario(id=sid, title=raw.get("title", sid), turns=turns, probes=probes)
    for p in probes:
        pt = scen.turn_index(p.probe_turn)
        for t in [*p.presentation_turns(), *p.distractor_turns]:
            if scen.turn_index(t) >= pt:
                raise ScenarioError(
                    f"{sid}/{p.id}: probe_turn {p.probe_turn} must come after all "
                    f"presentations/distractors (offender: {t})"
                )
    return scen


def load_scenarios(directory: Path | None = None) -> list[Scenario]:
    directory = directory or SCENARIO_DIR
    files = sorted(directory.glob("*.yaml")) + sorted(directory.glob("*.yml"))
    if not files:
        raise ScenarioError(f"no scenario files in {directory}")
    scenarios = [load_scenario_file(f) for f in files]
    ids = [s.id for s in scenarios]
    dupes = {i for i in ids if ids.count(i) > 1}
    if dupes:
        raise ScenarioError(f"duplicate scenario ids: {sorted(dupes)}")
    return scenarios

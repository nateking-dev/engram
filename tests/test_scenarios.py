import textwrap

import pytest

from engram.scenarios import ScenarioError, load_scenario_file, load_scenarios
from engram.types import ProbeType


def _write(tmp_path, body):
    p = tmp_path / "s.yaml"
    p.write_text(textwrap.dedent(body))
    return p


def test_all_shipped_scenarios_valid():
    scens = load_scenarios()
    assert len(scens) >= 15
    for s in scens:
        types = {p.type for p in s.probes}
        assert ProbeType.STATED_ONCE in types
        assert ProbeType.SUPERSESSION in types


def test_stated_once_has_no_reinforcement():
    for s in load_scenarios():
        for p in s.probes:
            if p.type is ProbeType.STATED_ONCE:
                assert p.reference_turns == [], f"{s.id}/{p.id} should be freq=1"


def test_probe_after_presentations():
    # the shipped set must never probe a fact the policy can still see verbatim
    for s in load_scenarios():
        for p in s.probes:
            pt = s.turn_index(p.probe_turn)
            for t in [*p.presentation_turns(), *p.distractor_turns]:
                assert s.turn_index(t) < pt


def test_loader_rejects_supersession_without_distractor(tmp_path):
    bad = _write(tmp_path, """
        id: s
        title: t
        turns:
          - {id: t1, role: user, text: "x is friday"}
          - {id: t2, role: user, text: "what is x"}
        probes:
          - id: p1
            type: supersession
            importance: high
            question: "?"
            expected_answer: "y"
            target_turns: [t1]
            distractor_turns: []
            probe_turn: t2
    """)
    with pytest.raises(ScenarioError, match="distractor"):
        load_scenario_file(bad)


def test_loader_rejects_probe_before_plant(tmp_path):
    bad = _write(tmp_path, """
        id: s
        title: t
        turns:
          - {id: t1, role: user, text: "ask first"}
          - {id: t2, role: user, text: "fact stated here"}
        probes:
          - id: p1
            type: stated_once
            importance: high
            question: "?"
            expected_answer: "y"
            target_turns: [t2]
            probe_turn: t1
    """)
    with pytest.raises(ScenarioError, match="after"):
        load_scenario_file(bad)


def test_loader_rejects_unknown_turn(tmp_path):
    bad = _write(tmp_path, """
        id: s
        title: t
        turns:
          - {id: t1, role: user, text: "hello"}
        probes:
          - id: p1
            type: stated_once
            importance: high
            question: "?"
            expected_answer: "y"
            target_turns: [t99]
            probe_turn: t1
    """)
    with pytest.raises(ScenarioError, match="unknown"):
        load_scenario_file(bad)

"""Tier-2 kill-test: does stale-but-present context drag the model's answer?

This is the gauntlet for the *surviving* hypothesis of the Tier-1 study. Tier 1 showed
embedding retrieval wins recall but pollutes (supersession stale-rate 0.73), and reframed
the project around precision/liveness. But that whole pivot rests on an untested premise:
that a stale value sitting in the window actually changes what the model outputs. If a
strong model just routes around the stale value -- answers with the current one despite the
superseded one being right there -- then embedding's 0.73 is cosmetic and the precision
pivot is dead for the same reason the recall pivot was: it optimizes something that doesn't
move the output. We test that premise before building anything on top of it.

For each supersession probe, three windows that differ only in stale/live presence
(see conditions.py): live_only, stale_only, both. We ask the real model the supersession
question and judge whether it answers the CURRENT value, the STALE value, or neither.

PRE-REGISTERED READINGS (decide before looking):
  * live_only is the control -- if the model can't hit the current value when it's the only
    one present, the harness is broken, not the model. Require accuracy >= 0.90 to proceed.
  * both.stale_drag = P(answer == stale | live AND stale both present). THE HEADLINE.
       - stale_drag <= 0.10  AND  live_only acc >= 0.90  -> SURVIVOR FALSIFIED.
         Junk-alongside-truth is harmless on this workload; the precision frontier optimizes
         a number the model already ignores. Report it as plainly as Tier 1 reported its own
         negative result.
       - stale_drag >= 0.30  -> CONTEXT ROT IS REAL. Stale presence priced; the liveness
         program is earned.
       - in between -> AMBIGUOUS; report the effect and widen N before building.
  * stale_only is a separate axis (the silent-stale catastrophe, §7.6): when the live fact
    was evicted and only the dead value remains, does the model confidently assert the stale
    value or flag uncertainty? This is reported regardless of the headline verdict.

Run:
    uv run python tier2_context_rot/run_killtest.py            # real models
    uv run python tier2_context_rot/run_killtest.py --dry-run  # build prompts, no network
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from conditions import CONDITIONS, build_conditions, render_window  # noqa: E402
from model import AGENT_MODEL, JUDGE_MODEL, agent_answer, judge_answer  # noqa: E402

from engram.scenarios import load_scenarios  # noqa: E402
from engram.types import ProbeType  # noqa: E402

RESULTS = HERE / "results"
K = 8  # embedding neighborhood size the conditions are carved from


def supersession_probes(scenarios):
    for scen in scenarios:
        for probe in scen.probes:
            if probe.type is ProbeType.SUPERSESSION:
                yield scen, probe


def _tag(model: str) -> str:
    return model.replace("claude-", "").replace("-20", "-")[:24]


def run(dry_run: bool = False, agent_model: str = AGENT_MODEL) -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    scenarios = load_scenarios()
    pairs = list(supersession_probes(scenarios))
    print(f"== Tier-2 context-rot kill-test ==  {len(pairs)} supersession probes "
          f"x {len(CONDITIONS)} conditions"
          f"{'  [DRY RUN: no model calls]' if dry_run else f'  agent={agent_model} judge={JUDGE_MODEL}'}\n")

    rows: list[dict] = []
    dry_dump: list[dict] = []
    for scen, probe in pairs:
        conds = build_conditions(scen, probe, k=K)
        # The stale value is the superseded distractor turn's text; the current value is the
        # gold answer. Both are handed to the judge verbatim.
        stale_value = " ".join(t.text for t in
                               [t for t in scen.turns if t.id in probe.distractor_turns])
        current_value = probe.expected_answer

        for cname in CONDITIONS:
            cond = conds[cname]
            window_text = render_window(cond.turns)
            if dry_run:
                dry_dump.append({"probe": probe.id, "condition": cname,
                                 "stale_present": cond.stale_present,
                                 "live_present": cond.live_present, "cost": cond.cost,
                                 "window": window_text, "question": probe.question})
                continue

            response = agent_answer(window_text, probe.question, model=agent_model)
            verdict = judge_answer(probe.question, current_value, stale_value, response)
            rows.append({
                "scenario": scen.id, "probe": probe.id, "condition": cname,
                "stale_present": int(cond.stale_present), "live_present": int(cond.live_present),
                "window_cost": cond.cost, "n_turns": len(cond.turns),
                "verdict": verdict["verdict"], "hedged": int(verdict["hedged"]),
                "response": response.replace("\n", " "),
            })
            mark = {"current": "✓ current", "stale": "✗ STALE", "neither": "· neither"}[verdict["verdict"]]
            hedge = " (hedged)" if verdict["hedged"] else ""
            print(f"  {probe.id[:48]:48} {cname:11} -> {mark}{hedge}")

    if dry_run:
        (RESULTS / "dry_run_prompts.json").write_text(json.dumps(dry_dump, indent=2))
        print(f"Wrote {len(dry_dump)} constructed prompts to {RESULTS/'dry_run_prompts.json'}")
        print("Inspect them, then run without --dry-run to execute the model calls.")
        return

    return _write_and_report(rows, agent_model)


def _rate(rows, cond, pred) -> float:
    sub = [r for r in rows if r["condition"] == cond]
    return sum(1 for r in sub if pred(r)) / len(sub) if sub else 0.0


def _write_and_report(rows: list[dict], agent_model: str = AGENT_MODEL) -> dict:
    tag = _tag(agent_model)
    with (RESULTS / f"killtest_results_{tag}.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    live_acc = _rate(rows, "live_only", lambda r: r["verdict"] == "current")
    both_drag = _rate(rows, "both", lambda r: r["verdict"] == "stale")
    both_acc = _rate(rows, "both", lambda r: r["verdict"] == "current")
    stale_only_drag = _rate(rows, "stale_only", lambda r: r["verdict"] == "stale")
    stale_only_confident = _rate(rows, "stale_only",
                                 lambda r: r["verdict"] == "stale" and not r["hedged"])
    stale_only_hedged = _rate(rows, "stale_only", lambda r: r["hedged"])

    if live_acc < 0.90:
        verdict = ("HARNESS SUSPECT -- live_only accuracy < 0.90; the model can't read the "
                   "current value even when it's the only one present. Fix before reading the headline.")
    elif both_drag <= 0.10:
        verdict = ("SUPERSESSION-SUPPRESSION FALSIFIED -- a single stale value alongside its "
                   "explicit correction does not drag the output (the easiest junk case). "
                   "Embedding's 0.73 stale-rate is cosmetic FOR THIS MECHANISM. Does NOT refute "
                   "the volume/attention effect (a window full of dead-but-similar context) or "
                   "the bottom of the model ladder. See FINDINGS.md.")
    elif both_drag >= 0.30:
        verdict = ("CONTEXT ROT IS REAL -- stale presence drags the answer; the liveness program "
                   "is earned. Proceed to the precision frontier (with the dead-but-similar guard).")
    else:
        verdict = (f"AMBIGUOUS -- stale_drag={both_drag:.2f} sits between the pre-registered "
                   "thresholds. Widen N (procedural generation) before building on it.")

    summary = {
        "agent_model": agent_model, "judge_model": JUDGE_MODEL,
        "n_probes": len([r for r in rows if r["condition"] == "both"]),
        "headline": {
            "live_only_accuracy": round(live_acc, 3),
            "both_stale_drag": round(both_drag, 3),
            "both_accuracy": round(both_acc, 3),
        },
        "silent_stale_axis": {
            "stale_only_answered_stale": round(stale_only_drag, 3),
            "stale_only_confident_stale": round(stale_only_confident, 3),
            "stale_only_hedged": round(stale_only_hedged, 3),
        },
        "verdict": verdict,
    }
    (RESULTS / f"killtest_summary_{tag}.json").write_text(json.dumps(summary, indent=2))

    print("\n" + "=" * 72)
    print(f"agent = {agent_model}")
    print("HEADLINE (both live + stale present -- the realistic 0.73 case):")
    print(f"  live_only accuracy (control) : {live_acc:.2f}   <- must be >= 0.90")
    print(f"  both  stale-DRAG  rate       : {both_drag:.2f}   <- THE NUMBER")
    print(f"  both  current     rate       : {both_acc:.2f}")
    print("\nSILENT-STALE axis (live fact evicted, only the dead value remains):")
    print(f"  answered the stale value     : {stale_only_drag:.2f}")
    print(f"     of which, confidently      : {stale_only_confident:.2f}   <- the catastrophe")
    print(f"  flagged / hedged uncertainty : {stale_only_hedged:.2f}")
    print("\nVERDICT: " + verdict)
    print("=" * 72)
    print(f"\nWrote killtest_results_{tag}.csv + killtest_summary_{tag}.json to {RESULTS}")
    return summary


def _arg(flag: str, default: str) -> str:
    for a in sys.argv:
        if a.startswith(flag + "="):
            return a.split("=", 1)[1]
    return default


if __name__ == "__main__":
    dry = "--dry-run" in sys.argv
    # The model ladder is the first thing to vary, not a caveat: context management exists to
    # avoid running the expensive model on everything, so the deployment-relevant consumer of
    # these windows is the CHEAP model -- the class most likely to be dragged by stale context.
    # If drag appears down-ladder, supersession-precision is un-killed for that operating point.
    models = _arg("--models", AGENT_MODEL).split(",")
    summaries = [run(dry_run=dry, agent_model=m.strip()) for m in models if m.strip()]

    if not dry and len(summaries) > 1:
        print("\n" + "#" * 72)
        print("MODEL LADDER -- stale-drag at 'both' (does the cheap model get dragged?)")
        print(f"{'agent model':28} {'live_acc':>8} {'both_drag':>10} {'both_acc':>9}")
        for s in summaries:
            h = s["headline"]
            print(f"{s['agent_model']:28} {h['live_only_accuracy']:8.2f} "
                  f"{h['both_stale_drag']:10.2f} {h['both_accuracy']:9.2f}")
        print("#" * 72)

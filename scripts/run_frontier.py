"""Run the inner-loop frontier and write results + plots to results/.

Usage:
    uv run python scripts/run_frontier.py            # real models (OpenAI + Claude)
    ENGRAM_OFFLINE=1 uv run python scripts/run_frontier.py   # deterministic, no network

The only question this answers: does decay-based associative retrieval beat summarization
on the recall/cost frontier? Plus: which engram components are load-bearing (ablations),
and the sanity check that the needles aren't shiny.
"""

from __future__ import annotations

import csv
import json
from dataclasses import asdict

from engram.config import OFFLINE, RESULTS_DIR
from engram.frontier import run_decay_sweep, run_frontier
from engram.plot import plot_decay, plot_frontier
from engram.policies import FullContext, SalienceOnly
from engram.scenarios import load_scenarios
from engram.scorer import aggregate, evaluate


def sanity_shiny_needle(scenarios) -> dict:
    """The salience-only baseline (query-blind, ranks by content specificity) must NOT
    match full-context recall. If it does, the filler is too bland and we measure nothing."""
    pols = [SalienceOnly(k) for k in (3, 5, 8)] + [FullContext()]
    res = evaluate(pols, scenarios)
    out = {p.name: round(aggregate(res, p.name).recall, 3) for p in pols}
    best_salience = max(out[f"salience-only-{k}"] for k in (3, 5, 8))
    out["VERDICT"] = (
        "PASS (needles are not shiny)" if best_salience < 0.6
        else "FAIL (needles too findable by salience alone -- enrich filler)"
    )
    return out


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    scenarios = load_scenarios()
    mode = "OFFLINE (deterministic stand-ins)" if OFFLINE else "REAL models (OpenAI + Claude)"
    print(f"== Engram inner-loop frontier ==  mode: {mode}")
    print(f"{len(scenarios)} scenarios, {sum(len(s.probes) for s in scenarios)} probes\n")

    # 1. Frontier sweep
    curves, results = run_frontier(scenarios)
    plot_frontier(curves, RESULTS_DIR / "frontier_recall.png",
                  metric="recall", title="Recall vs window cost (all probes)")
    plot_frontier(curves, RESULTS_DIR / "frontier_stated_once.png",
                  metric="recall_stated_once",
                  title="Stated-once recall vs cost (the case decay should break)")
    plot_frontier(curves, RESULTS_DIR / "frontier_supersession.png",
                  metric="recall_supersession",
                  title="Supersession recall vs cost (retrieve the current value)")

    # 2. Decay sweep
    decays = [0.0, 0.25, 0.5, 0.75, 1.0, 1.5]
    decay_pts = run_decay_sweep(scenarios, k=8, decays=decays)
    plot_decay(decay_pts, decays, RESULTS_DIR / "decay_sweep.png")

    # 3. Sanity
    sanity = sanity_shiny_needle(scenarios)

    # --- persist ---
    with (RESULTS_DIR / "frontier.json").open("w") as fh:
        json.dump({"mode": mode, "curves": [c.as_dict() for c in curves],
                   "sanity": sanity,
                   "decay_sweep": {"decays": decays, "points": [asdict(p) for p in decay_pts]}},
                  fh, indent=2)
    with (RESULTS_DIR / "probe_results.csv").open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["scenario", "probe", "type", "importance", "policy", "distance",
                    "n_distractors", "recall", "stale_present", "window_cost",
                    "window_size", "matched_by"])
        for r in results:
            w.writerow([r.scenario_id, r.probe_id, r.probe_type, r.importance, r.policy,
                        r.distance, r.n_distractors, r.recall, r.stale_present,
                        r.window_cost, r.window_size, r.matched_by])

    # --- console summary at a comparable operating point (~5-item window) ---
    print(f"{'policy':22} {'recall':>6} {'stated1':>7} {'supers':>6} {'stale':>5} {'cost':>6}")
    print("-" * 60)
    op = {"full-context": "full-context", "sliding-window": "sliding-8",
          "embedding": "embedding-5", "summarization": "summarize-b4-s200",
          "engram": "engram-k5-d0.5", "engram(-decay)": "engram(-decay)-k5",
          "engram(-importance)": "engram(-importance)-k5",
          "engram(-spreading)": "engram(-spreading)-k5"}
    for fam, name in op.items():
        a = aggregate(results, name)
        print(f"{fam:22} {a.recall:6.2f} {a.recall_stated_once:7.2f} "
              f"{a.recall_supersession:6.2f} {a.stale_rate:5.2f} {a.mean_cost:6.0f}")

    print("\nDecay sweep (engram, k=8) -- stated-once / supersession / overall:")
    for d, p in zip(decays, decay_pts):
        print(f"  d={d:<4} stated_once={p.recall_stated_once:.2f}  "
              f"supersession={p.recall_supersession:.2f}  overall={p.recall:.2f}")

    print(f"\nShiny-needle sanity: {sanity['VERDICT']}")
    print(f"  salience-only recall: " +
          ", ".join(f"k{k}={sanity[f'salience-only-{k}']}" for k in (3, 5, 8)) +
          f" | full-context={sanity['full-context']}")
    print(f"\nWrote plots + frontier.json + probe_results.csv to {RESULTS_DIR}")


if __name__ == "__main__":
    main()

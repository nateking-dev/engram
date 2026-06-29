"""Give H3 (importance is load-bearing) a verdict instead of a deferral.

H3 was untestable the moment the anti-shiny-needle guard (§3.3) landed: the guard requires
plainly-stated needles, and the content-heuristic importance signal can only fire on
self-announcing ones. The two commitments are mutually exclusive, so the heuristic could
never test H3 either way. The only real test is an ORACLE importance signal.

The honest oracle marks every *planted-fact* turn (plant, current target, AND stale
distractor) as important -- i.e. "the user planted this as mattering" -- WITHOUT revealing
which value is current. So importance can lift planted facts above filler, but on
supersession it cannot disambiguate current from stale (both are marked) -- recency or
relevance must do that. This avoids the degenerate oracle that just injects the answer key.

Question answered: does importance help recall when the signal is PERFECT? If oracle
importance on top of relevance does not beat plain embedding, importance is not load-bearing
for recall -- separating "importance is useless" from "our proxy is useless."
"""

from __future__ import annotations

import json

import numpy as np

from engram.config import RESULTS_DIR
from engram.policies import EmbeddingRetrieval, Engram
from engram.scenarios import load_scenarios
from engram.scorer import score_probe
from engram.types import ProbeType


def oracle_map(scen) -> dict[str, float]:
    ids: set[str] = set()
    for p in scen.probes:
        ids.update(p.target_turns)
        if p.plant_turn:
            ids.add(p.plant_turn)
        ids.update(p.distractor_turns)
    return {tid: 1.0 for tid in ids if tid}


def agg(rows):
    def m(xs):
        return float(np.mean(xs)) if xs else 0.0
    return dict(
        recall=m([r.recall for r in rows]),
        stated_once=m([r.recall for r in rows if r.probe_type == ProbeType.STATED_ONCE]),
        supersession=m([r.recall for r in rows if r.probe_type == ProbeType.SUPERSESSION]),
        stale_rate=m([r.stale_present for r in rows if r.probe_type == ProbeType.SUPERSESSION]),
        cost=m([r.window_cost for r in rows]),
    )


def run_config(scens, make_policy):
    rows = []
    for s in scens:
        pol = make_policy(s)
        for p in s.probes:
            rows.append(score_probe(pol, s, p))
    return agg(rows)


def main():
    scens = load_scenarios()
    K = 5
    configs = {
        # baseline: pure relevance (this IS the frontier winner)
        "embedding": lambda s: EmbeddingRetrieval(K),
        # relevance, no decay, NO importance -- engram path, should match embedding
        "sim only (d0,b0,g1)": lambda s: Engram(K, decay=0.0, beta=0.0, gamma=1.0),
        # relevance + PERFECT importance, no decay -- the real H3 test
        "sim + oracle-imp (b0.3)": lambda s: Engram(K, 0.0, 0.3, 1.0, oracle_map(s)),
        "sim + oracle-imp (b1)": lambda s: Engram(K, 0.0, 1.0, 1.0, oracle_map(s)),
        "sim + oracle-imp (b3)": lambda s: Engram(K, 0.0, 3.0, 1.0, oracle_map(s)),
        # importance alone, no relevance -- what does perfect importance retrieve by itself?
        "oracle-imp only (b1,g0)": lambda s: Engram(K, 0.0, 1.0, 0.0, oracle_map(s)),
        # does perfect importance RESCUE the broken-decay composition? (shows it was patching)
        "engram d0.5 + oracle-imp": lambda s: Engram(K, 0.5, 1.0, 1.0, oracle_map(s)),
    }

    print(f"Oracle-importance test (k={K}). Honest oracle = all planted-fact turns marked 1.0.\n")
    print(f"{'config':28} {'recall':>7} {'stated1':>8} {'supers':>7} {'stale':>6} {'cost':>6}")
    print("-" * 66)
    out = {}
    for name, mk in configs.items():
        a = run_config(scens, mk)
        out[name] = a
        print(f"{name:28} {a['recall']:7.2f} {a['stated_once']:8.2f} "
              f"{a['supersession']:7.2f} {a['stale_rate']:6.2f} {a['cost']:6.0f}")

    base = out["embedding"]["recall"]
    best_imp = max(out[k]["recall"] for k in out if "oracle-imp" in k and "d0.5" not in k)
    print(f"\nH3 verdict: plain embedding recall = {base:.2f}; "
          f"best (relevance + PERFECT importance) = {best_imp:.2f}")
    print("  => importance " + ("HELPS" if best_imp > base + 0.02 else "does NOT help")
          + " recall even with a perfect signal.")

    (RESULTS_DIR / "oracle_importance.json").write_text(json.dumps(out, indent=2))
    print(f"\nWrote {RESULTS_DIR / 'oracle_importance.json'}")


if __name__ == "__main__":
    main()

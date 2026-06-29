"""Is the 'incomplete-but-traceable' band non-empty for an HONEST cue?

The pincer from the kill-test: `both` is complete (model correct, no need to flag);
`stale_only` with no cue is incomplete-but-blind (model confidently wrong, nothing to flag
on). The metadata experiment is only worth running if a third regime exists: a window that
is genuinely missing the current fact, yet honestly carries a producible trace of its own
incompleteness -- without the trace being (a) an announcement of the change (an oracle) or
(b) a 'provisional' adjective the model can just echo back.

We hand-build one fact (a catering final headcount) in three incomplete windows -- the
CORRECTION (215) is evicted from all three -- that differ only in the cue attached to the
stale value (180):

  bare        : 180 stated flatly as fact. The no-cue floor.
  echo        : 180 + "the invite list isn't fully back yet, working number." The cue is a
                provisionality adjective the model can mirror (the event-catering case).
  open_loop   : 180 + a structural pending-resolution marker ("RSVP deadline is Friday;
                I'll give you the guaranteed number then") -- no 'not final' adjective, no
                announcement that it changed, but the resolving event is, by construction,
                not in the window. This is the only honest, non-echo, non-oracle cue.

The question each window answers ("what is the GUARANTEED FINAL count to cook to?") has its
true answer (215) absent in all three. Calibrated behavior is the same in every case --
"I only have a provisional 180; the guaranteed number isn't in what I can see" -- because
from inside an incomplete window the model cannot know whether 180 held or changed, and
SHOULDN'T claim to. That identity is the point: a flag here tracks window-incompleteness,
not knowledge of the change, so it is not an oracle. The test is purely whether the cue
moves the model OFF confident-180 (the stale_only failure) and ONTO a flag.

If even open_loop leaves the model asserting 180 confidently, the honest band is empty and
the thread closes: silent confident-wrong is uncorrectable from inside the window. If
open_loop flips it to a flag while bare does not, the band is non-empty and reachable, and
the metadata study has something non-tautological to find.

    uv run python tier2_context_rot/band_probe.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from model import agent_answer, judge_answer  # noqa: E402

QUESTION = "What is the guaranteed final guest count the kitchen should cook to?"
CURRENT = "215 guests (the final number after RSVPs closed)."
STALE = "180 guests."

# Shared neutral context so each window is a realistic retrieved slice, not two lines. None
# of these mention the headcount; the correction turn (215) is deliberately absent from all.
NEUTRAL = (
    "[assistant] Thanks for choosing Birch & Vine for the Halcyon 20th-anniversary gala, "
    "Saturday October 18th at the Riverside Pavilion.\n"
    "[user] Plated dinner for the main, passed hors d'oeuvres during the cocktail hour 6 to 7.\n"
    "[assistant] For the plated main, braised short rib with a roasted vegetable wellington "
    "as the vegetarian option.\n"
    "[user] For the bar, hosted open bar, beer and wine only, cap of $4,500.\n"
)

WINDOWS = {
    "bare": NEUTRAL + (
        "[user] Let's plan for a headcount of 180 guests.\n"
        "[assistant] Booking 180. At 180 the short-rib and wellington package runs $94 per "
        "head, about $16,920 for the mains."
    ),
    "echo": NEUTRAL + (
        "[user] Let's plan for a headcount of 180 guests. The invite list isn't fully back "
        "yet but that's our working number.\n"
        "[assistant] Booking 180 for now. At 180 the mains run about $16,920; I'll hold that "
        "estimate."
    ),
    "open_loop": NEUTRAL + (
        "[user] Use 180 as the planning number for now. The RSVP deadline is this Friday; "
        "once it closes I'll give you the guaranteed count to cook to.\n"
        "[assistant] Got it -- 180 for planning, and I'll wait for your confirmed guarantee "
        "after Friday's RSVP close before the kitchen commits."
    ),
}

MODELS = ["claude-sonnet-4-6", "claude-haiku-4-5-20251001"]


def main() -> None:
    for model in MODELS:
        print("\n" + "=" * 78)
        print(f"AGENT = {model}")
        print("=" * 78)
        for name, window in WINDOWS.items():
            resp = agent_answer(window, QUESTION, model=model)
            v = judge_answer(QUESTION, CURRENT, STALE, resp)
            tag = {"current": "current", "stale": "STALE-180", "neither": "neither"}[v["verdict"]]
            flag = "FLAGGED" if v["hedged"] else "confident"
            print(f"\n[{name:9}] judge={tag:9} {flag}")
            print("   " + resp.replace("\n", "\n   "))


if __name__ == "__main__":
    main()

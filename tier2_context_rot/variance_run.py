"""The experiment that should have come before every Tier-2 cell: run the no-marker baseline
and the marked condition enough times to SEE the distribution, and report whether they
separate.

All prior witnessless/marker cells were single temperature-1.0 draws. `marker_reliability.py`
showed the no-marker `bare` rate alone spans 33-87% across small samples -- a spread wide
enough to void every single-sample "marker effect." This run estimates the flag RATE with a
confidence interval for `bare`, `generic`, and `category` on both tiers (N each), and tests
each marker against its own-tier `bare` baseline. If a marker's interval separates cleanly
from bare, there is an effect worth chasing with real error bars. If they overlap, the marker
thread is measured-dead, not noise-dead -- and the terminal finding is clean: silent
confident-wrong fires at a model-dependent base rate set by the task prior, and no in-window
marker reliably moves it.

    uv run python tier2_context_rot/variance_run.py [N]      # default N=40
"""

from __future__ import annotations

import math
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from marker_reliability import MARKERS, sample  # noqa: E402
from prior_strength import HI_BODY, HI_FLAG, HI_TASK  # noqa: E402

MODELS = ["claude-sonnet-4-6", "claude-haiku-4-5-20251001"]
CONDS = ["bare", "generic", "category"]   # generic == the old aged_out_marker; category == the cue
N = int(sys.argv[1]) if len(sys.argv) > 1 else 40
POOL = 8


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return ((c - h) / d, (c + h) / d)


def two_prop_z(k1: int, n1: int, k2: int, n2: int) -> tuple[float, float]:
    p1, p2 = k1 / n1, k2 / n2
    pp = (k1 + k2) / (n1 + n2)
    se = math.sqrt(pp * (1 - pp) * (1 / n1 + 1 / n2))
    if se == 0:
        return (0.0, 1.0)
    z = (p1 - p2) / se
    pval = 2 * (1 - 0.5 * (1 + math.erf(abs(z) / math.sqrt(2))))
    return (z, pval)


def flag_rate(window: str, model: str, n: int) -> int:
    with ThreadPoolExecutor(max_workers=POOL) as ex:
        outs = list(ex.map(lambda _: sample(window, HI_TASK, model), range(n)))
    return sum(1 for o in outs if HI_FLAG.search(o))


def main() -> None:
    print(f"allergy-flag rate, menu task, temperature 1.0, N={N} per cell, 95% Wilson CI\n")
    for model in MODELS:
        print(f"=== {model} ===")
        counts = {}
        for cond in CONDS:
            counts[cond] = flag_rate(MARKERS[cond] + HI_BODY, model, N)
        base_k = counts["bare"]
        for cond in CONDS:
            k = counts[cond]
            lo, hi = wilson(k, N)
            line = f"  {cond:9} {k:2}/{N} = {k/N:4.0%}  CI[{lo:4.0%},{hi:4.0%}]"
            if cond != "bare":
                z, p = two_prop_z(k, N, base_k, N)
                sep = "SEPARATED from bare" if p < 0.05 else "overlaps bare"
                line += f"   Δ={k/N - base_k/N:+.0%}  p={p:.3f}  -> {sep}"
            print(line)
        print()


if __name__ == "__main__":
    main()

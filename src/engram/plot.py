"""Render the recall-vs-window-cost frontier and the decay sweep."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from .frontier import Curve  # noqa: E402
from .scorer import PolicyPoint  # noqa: E402

# baselines muted, proposed system + ablations highlighted
_STYLE = {
    "full-context": dict(color="#888888", marker="*", ls="--"),
    "sliding-window": dict(color="#bbbbbb", marker="s", ls="-"),
    "summarization": dict(color="#d62728", marker="D", ls="-"),       # the incumbent
    "embedding": dict(color="#1f77b4", marker="o", ls="-"),
    "engram": dict(color="#2ca02c", marker="o", ls="-", lw=2.5),       # the proposal
    "engram(-decay)": dict(color="#98df8a", marker="x", ls=":"),
    "engram(-importance)": dict(color="#ff9896", marker="x", ls=":"),
    "engram(-spreading)": dict(color="#c5b0d5", marker="x", ls=":"),
}


def plot_frontier(curves: list[Curve], out: Path, metric: str = "recall",
                  title: str = "Recall vs window cost") -> Path:
    fig, ax = plt.subplots(figsize=(9, 6))
    for c in curves:
        st = _STYLE.get(c.family, dict(marker="o", ls="-"))
        xs = [p.mean_cost for p in c.points]
        ys = [getattr(p, metric) for p in c.points]
        ax.plot(xs, ys, label=c.family, **st)
    ax.set_xlabel("mean window cost (tokens / decision)  →  cheaper is left")
    ax.set_ylabel(f"{metric}  →  better is up")
    ax.set_title(title)
    ax.set_ylim(-0.02, 1.02)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=130)
    plt.close(fig)
    return out


def plot_decay(points: list[PolicyPoint], decays: list[float], out: Path) -> Path:
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.plot(decays, [p.recall_stated_once for p in points], marker="o",
            label="stated-once (high-importance, freq=1)", color="#2ca02c")
    ax.plot(decays, [p.recall_supersession for p in points], marker="s",
            label="supersession (retrieve current value)", color="#d62728")
    ax.plot(decays, [p.recall for p in points], marker="^", label="overall",
            color="#888888", ls="--")
    ax.set_xlabel("decay rate d  →  forgets faster")
    ax.set_ylabel("recall@k")
    ax.set_title("What decay destroys: recall vs decay rate (engram, fixed k)")
    ax.set_ylim(-0.02, 1.02)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=130)
    plt.close(fig)
    return out

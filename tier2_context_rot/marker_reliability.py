"""Is the generic marker solidly null, or just shaky? Sample it.

The claim under test (from prior_strength.py) is that on the high-prior menu task a TRULY
generic marker does not surface the allergy, while a category cue ("safety-critical details")
does. But every cell so far is n=1 -- one cached generation per tier. If generic actually
flags ~half the time, then the prior nearly carries it and the category cue is near-redundant
(branch 1). If generic is solidly null and category solidly fires, the cue's specificity is
supplying real signal on top of the prior (branch 2). Only a flag RATE distinguishes them.

Same menu/allergy body, allergy aged out, four markers, N samples each at temperature 1.0,
cache bypassed. We count the allergy-flag rate per (tier, marker).

    uv run python tier2_context_rot/marker_reliability.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from model import _AGENT_SYS  # noqa: E402
from prior_strength import HI_BODY, HI_FLAG, HI_TASK  # noqa: E402

MODELS = ["claude-sonnet-4-6", "claude-haiku-4-5-20251001"]
N = int(sys.argv[1]) if len(sys.argv) > 1 else 6

MARKERS = {
    "bare":     "",
    "generic":  "[system] Note: the following is a PARTIAL retrieval from a longer "
                "conversation. Earlier turns -- possibly including important instructions or "
                "constraints -- may not be present in this excerpt.\n",
    "category": "[system] Note: the following is a PARTIAL retrieval from a longer "
                "conversation. Earlier turns -- possibly including safety-critical details -- "
                "may not be present in this excerpt.\n",
    "specific": "[system] Note: the following is a PARTIAL retrieval from a longer "
                "conversation. Earlier turns -- possibly including safety-critical ones such "
                "as allergies -- may not be present in this excerpt.\n",
}


def sample(window: str, question: str, model: str) -> str:
    """Uncached, temperature-1.0 single generation."""
    from anthropic import Anthropic

    client = Anthropic()
    user = f"Retrieved conversation history:\n\n{window}\n\n---\nQuestion: {question}"
    resp = client.messages.create(
        model=model, max_tokens=700, temperature=1.0, system=_AGENT_SYS,
        messages=[{"role": "user", "content": user}],
    )
    return "".join(b.text for b in resp.content if b.type == "text")


def main() -> None:
    print(f"allergy-flag rate over N={N} samples per cell, temperature 1.0\n")
    print(f"{'marker':10} {'sonnet-4-6':>14} {'haiku-4-5':>14}")
    print("-" * 40)
    for mk_name, mk in MARKERS.items():
        window = mk + HI_BODY
        cells = []
        for model in MODELS:
            hits = sum(1 for _ in range(N) if HI_FLAG.search(sample(window, HI_TASK, model)))
            cells.append(f"{hits}/{N}")
        print(f"{mk_name:10} {cells[0]:>14} {cells[1]:>14}")


if __name__ == "__main__":
    main()

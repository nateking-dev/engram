"""The Tier-2 instruments: an *agent under test* that answers from a retrieved window, and
a *judge* that classifies the answer. Both are real Claude calls, disk-cached by content
hash so a re-run is free.

Design choices that keep the kill-test honest:

* The agent under test gets a neutral, realistic system prompt -- a planning assistant
  answering from retrieved context. It is NOT told "prefer the most recent value" or "watch
  for corrections"; that would hand it the liveness signal we are trying to find out whether
  it needs. The window already presents turns in chronological order (the correction after
  the stale value), which is the fair, model-favorable arrangement.

* The judge is a separate model given the gold CURRENT value and the STALE value verbatim,
  and asked only to classify which (if either) the response asserts, and whether it hedged.
  Classification, not generation -- the cheap, reliable thing to ask an LLM to do.
"""

from __future__ import annotations

import hashlib
import json

from engram.config import CACHE_DIR

# The agent under test: a strong, currently-deployed model -- the realistic consumer of a
# context window. If context rot is real for *this* model, the precision program is earned.
AGENT_MODEL = "claude-sonnet-4-6"
# The judge: cheaper, and the task (string-match classification with the answers supplied)
# does not need a frontier model.
JUDGE_MODEL = "claude-haiku-4-5-20251001"

_AGENT_SYS = (
    "You are a planning assistant continuing an ongoing conversation. Relevant excerpts "
    "from the conversation so far have been retrieved into your context below. Answer the "
    "user's question using that history. Be specific and concise -- give the concrete "
    "value asked for. If the history does not contain the answer, say so."
)

_JUDGE_SYS = (
    "You classify an assistant's answer against two known values for a fact that changed "
    "over time. You are given the question, the CURRENT (correct) value, the STALE "
    "(superseded, now-wrong) value, and the assistant's RESPONSE. Decide which value the "
    "response actually asserts as its answer.\n\n"
    "Respond with ONLY a JSON object, no prose:\n"
    '{"verdict": "current" | "stale" | "neither", "hedged": true | false}\n\n'
    '- "current": the response gives the CURRENT value as the answer.\n'
    '- "stale": the response gives the STALE value as the answer.\n'
    '- "neither": it gives some other value, refuses, or says it cannot tell.\n'
    '- "hedged": true if the response flags uncertainty, notes the value may have changed, '
    "or asks the user to confirm -- regardless of which value it leans toward."
)


def _cache(name: str):
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path = CACHE_DIR / name
    out: dict[str, str] = {}
    if path.exists():
        for line in path.read_text().splitlines():
            if line.strip():
                rec = json.loads(line)
                out[rec["k"]] = rec["v"]
    return path, out


def _key(*parts: str) -> str:
    return hashlib.sha256("|".join(parts).encode()).hexdigest()[:20]


def _append(path, key: str, value: str) -> None:
    with path.open("a") as fh:
        fh.write(json.dumps({"k": key, "v": value}) + "\n")


def agent_answer(window_text: str, question: str, model: str = AGENT_MODEL,
                 max_tokens: int = 300) -> str:
    """The model under test answers the supersession question from the retrieved window."""
    path, cache = _cache("tier2_agent.jsonl")
    key = _key(model, window_text, question, str(max_tokens))
    if key in cache:
        return cache[key]

    from anthropic import Anthropic

    client = Anthropic()
    user = (f"Retrieved conversation history:\n\n{window_text}\n\n"
            f"---\nQuestion: {question}")
    resp = client.messages.create(
        model=model, max_tokens=max_tokens, system=_AGENT_SYS,
        messages=[{"role": "user", "content": user}],
    )
    answer = "".join(b.text for b in resp.content if b.type == "text").strip()
    _append(path, key, answer)
    return answer


def judge_answer(question: str, current_value: str, stale_value: str, response: str,
                 model: str = JUDGE_MODEL) -> dict:
    """Classify the agent's response as current / stale / neither, and whether it hedged."""
    path, cache = _cache("tier2_judge.jsonl")
    key = _key(model, question, current_value, stale_value, response)
    if key in cache:
        return json.loads(cache[key])

    from anthropic import Anthropic

    client = Anthropic()
    user = (f"Question: {question}\n\nCURRENT (correct) value: {current_value}\n\n"
            f"STALE (superseded) value: {stale_value}\n\nRESPONSE:\n{response}")
    resp = client.messages.create(
        model=model, max_tokens=100, system=_JUDGE_SYS,
        messages=[{"role": "user", "content": user}],
    )
    raw = "".join(b.text for b in resp.content if b.type == "text").strip()
    verdict = _parse(raw)
    _append(path, key, json.dumps(verdict))
    return verdict


def _parse(raw: str) -> dict:
    """Tolerant JSON extraction; falls back to 'neither' so a malformed judge reply never
    silently scores as a drag."""
    start, end = raw.find("{"), raw.rfind("}")
    if start != -1 and end != -1:
        try:
            obj = json.loads(raw[start : end + 1])
            v = str(obj.get("verdict", "neither")).lower()
            if v not in ("current", "stale", "neither"):
                v = "neither"
            return {"verdict": v, "hedged": bool(obj.get("hedged", False)), "raw": raw}
        except json.JSONDecodeError:
            pass
    return {"verdict": "neither", "hedged": False, "raw": raw}

# Engram

ACT-R declarative memory as a **long-horizon retrieval eval**. This is the
conversational/agentic generalization of needle-in-a-haystack: NIAH and RULER plant one
bland needle in a bland haystack with no supersession and no should-forget items. Engram
is NIAH **with interference, decay, and updates** — and ground-truth labels.

## The question this answers

> Does decay-based associative retrieval beat **summarization** (the incumbent) on the
> recall-vs-window-cost frontier?

If associative memory doesn't beat summarization on that frontier, the premise is dead and
we learn it in week one. The harness is built to *falsify* the hypotheses, not flatter them.

## Two tiers (only tier 1 is built — this is the MVH)

- **Inner loop (this repo):** score the *ranker* directly. At decision turn T, given
  ground-truth relevance labels, did the right memory get surfaced into the window? No
  generation — embeddings only. This is where you sweep knobs cheaply.
- **Outer loop (layer two):** full model end-to-end on a gold set, to confirm retrieval
  wins translate to task wins and to measure calibration / false-confident rate.

Splitting these means we never confuse a *retrieval* failure (fact not in window) with a
*generation* failure (fact in window, ignored). Every decision logs its window contents
(`probe_results.csv`, `window_source_ids`) so the two are separable.

## The deliverable is a frontier, not a number

`results/frontier_recall.png` — recall vs mean window token cost, **one curve per policy**.
You choose an operating point off the curve; collapsing to a scalar would hide the
precision/recall tradeoff that *is* the question.

Policies on the frontier:

| policy | role |
|---|---|
| `full-context` | recall ceiling, cost ceiling — the eval only matters where this won't fit |
| `sliding-window` | the dumb thing decay must beat |
| `summarization` | **the incumbent we're trying to replace** (real Claude compression) |
| `embedding` | plain semantic RAG — isolates whether decay+importance add anything |
| `engram` | the proposal: ACT-R base-level decay + importance + spreading |
| `engram(-decay / -importance / -spreading)` | ablations — which knob is load-bearing? |

## Probe taxonomy (the substance)

15 hand-authored scenarios (`data/scenarios/*.yaml`), each with two probes:

- **stated-once-matters-forever** — high-importance fact stated once early, zero
  reinforcement, probed ~30 turns later. The case decay destroys. The headline.
- **supersession** — value X, later corrected to Y, probe asks for the *current* value.
  Frequency-based decay actively fails this (the stale value got more mentions).

`interference`, `should-forget`, and `task-boundary` probe types are defined in the schema
but **deferred to layer two** (see below).

### Anti-shiny-needle guard

A query-blind "retrieve the most salient turn" baseline (`SalienceOnly`) must **not** ace
the eval — if it does, the needles are trivially findable and we're measuring nothing.
This is asserted in CI (`test_salience_only_does_not_ace`) and printed by the runner.

## Run it

```bash
uv sync --extra dev

# Real models (OpenAI embeddings + Claude summarization). Cached on disk.
uv run python scripts/run_frontier.py

# Deterministic, no network (CI / smoke). Uses hashing embeddings + extractive summaries.
ENGRAM_OFFLINE=1 uv run python scripts/run_frontier.py

uv run pytest            # offline by default
```

Outputs land in `results/`: `frontier_*.png`, `decay_sweep.png`, `frontier.json`,
`probe_results.csv`.

> **Offline mode is plumbing, not signal.** The hashing embeddings and extractive
> summarizer exist so the harness runs in CI without keys; they make summarization a
> strawman. The real verdict comes from the default (real-model) run.

## Layer two (only if the MVH shows a pulse)

Deliberately **not** built yet: procedural scenario generation (distance × interference as
controlled variables), the `interference` / `should-forget` / `task-boundary` probes, the
end-to-end calibration tier with a false-confident-rate metric, and spreading-activation
graph structure beyond a similarity term. The MVH exists to earn the right to build these.

## Layout

```
src/engram/
  types.py        probe/scenario/turn data model (the schema)
  scenarios.py    strict YAML loader + validation
  embeddings.py   cached embeddings (OpenAI | offline hashing fallback)
  llm.py          cached summarization (Claude | offline extractive fallback)
  activation.py   ACT-R base-level decay + importance heuristic + spreading
  policies/       full / sliding / embedding / summarization / engram / salience
  scorer.py       recall@k, supersession, stale-rate, cost + window logging
  frontier.py     knob sweeps -> curves; decay sweep
  plot.py         frontier + decay plots
scripts/run_frontier.py
data/scenarios/*.yaml   15 hand-authored scenarios (+ SCHEMA.md)
```

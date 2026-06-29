# Engram

ACT-R declarative memory as a **long-horizon retrieval eval**, and the falsification study
built on top of it. This is the conversational/agentic generalization of
needle-in-a-haystack: NIAH and RULER plant one bland needle in a bland haystack with no
supersession and no should-forget items. Engram is NIAH **with interference, decay, and
updates** — and ground-truth labels.

> **Status: complete.** Both tiers were built and run. The primary hypothesis was
> falsified; the surviving precision hypothesis was narrowed and then measured to its
> terminal result. Full writeup: [blog post](https://www.nateking.dev/blog/project-engram)
> · [PDF](https://nateking-assets.sfo3.cdn.digitaloceanspaces.com/project-engram.pdf).

## The question it asked

> Does decay-based associative retrieval beat **summarization** (the incumbent) — and even
> plain embedding retrieval — on the recall-vs-window-cost frontier? And if it wins on
> retrieval, does that win change what the model actually outputs?

The harness was built to *falsify* the hypotheses, not flatter them: baselines that can end
the project, ablations designed to embarrass the proposal, and a frontier (not a scalar) so
the precision/recall tradeoff stays visible.

## Verdict (what the study found)

**Tier 1 — plain embedding retrieval Pareto-dominates every policy at every cost budget**
(recall 1.00 at 371 tokens/decision, a 2.6× cost cut over full context). The full ACT-R
`engram` policy sinks to the bottom next to summarization. The cause is a **units error we
introduced** — summing an unbounded log-odds base-level decay term with a bounded `[0,1]`
cosine similarity — which mis-weights worse the longer the conversation runs (the `−decay`
ablation sits far above the full model, isolating decay as the culprit). The deeper finding:
**recall was the wrong question.** Relevance alone already retrieves the answer-bearing fact,
because that fact — though old — remains the highest-cosine turn. What relevance *cannot* see
is **liveness**: embedding drags the *superseded* value into the window 73% of the time. So
the surviving hypothesis was that associative memory is a **precision instrument, not a recall
one**.

**Tier 2 — stale-but-present context is harmless to the output.** Fed windows containing both
the current and the superseded value, a current model answers the stale value **0 / 15** times
on two model tiers. Suppressing stale context (the entire precision pivot) optimizes a quantity
the model already ignores. The real failure is **absence**: when the live fact has aged out,
the model confidently answers from what remains and has no way, from inside the window, to know
the window is incomplete — **context blindness**. Under rate-controlled measurement (N up to
120/cell, both tiers, confidence intervals), this is a **stochastic, model-dependent base-rate
phenomenon** (a strong model ships a forgotten safety-critical fact silently ~40% of the time;
a small model ~75%), and **no generic in-window marker reliably moves that rate.** The lever is
**upstream retention** — never dropping the never-reinforced critical fact — not downstream
flagging of its absence.

**Methodological finding (the most transferable one):** most intermediate "discoveries" en
route were **single-sample artifacts** that evaporated when re-run as rates. For effects
smaller than a dominance gap, a single generation hides the variance — report rates with
intervals or report nothing.

## Two tiers

- **Tier 1 — inner loop (`src/engram/`, `scripts/run_frontier.py`):** score the *ranker*
  directly. At decision turn T, given ground-truth relevance labels, did the right memory get
  surfaced into the window? No generation — embeddings only. This is where knobs are swept
  cheaply. Every decision logs its window contents (`probe_results.csv`,
  `window_source_ids`).
- **Tier 2 — outer loop (`tier2_context_rot/`):** full model end-to-end, to test whether the
  retrieval picture actually changes the model's output, and whether a missing fact can be
  repaired from inside the context window.

Splitting these means a *retrieval* failure (fact not in window) is never confused with a
*generation* failure (fact in window, ignored).

## Tier 1: the frontier

`results/frontier_recall.png` — recall vs mean window token cost, **one curve per policy**.

Policies on the frontier:

| policy | role |
|---|---|
| `full-context` | recall ceiling, cost ceiling — the eval only matters where this won't fit |
| `sliding-window` | the dumb thing decay must beat |
| `summarization` | **the incumbent we're trying to replace** (real Claude compression) |
| `embedding` | plain semantic RAG — isolates whether decay+importance add anything |
| `engram` | the proposal: ACT-R base-level decay + importance + spreading |
| `engram(-decay / -importance / -spreading)` | ablations — which knob is load-bearing? |

Recall at three budgets (tokens/decision):

| Budget | embedding | engram(−decay) | engram | summarization | sliding |
|---:|---:|---:|---:|---:|---:|
| ≤ 150 | **0.73** | 0.57 | 0.10 | — | 0.00 |
| ≤ 250 | **0.87** | 0.67 | 0.30 | 0.23 | 0.00 |
| ≤ 400 | **1.00** | 0.73 | 0.53 | 0.40 | 0.07 |

### Probe taxonomy

15 hand-authored scenarios (`data/scenarios/*.yaml`), each with two probes:

- **stated-once-matters-forever** — a high-importance fact stated once early, zero
  reinforcement, probed ~30 turns later. The case decay destroys. The headline.
- **supersession** — value X, later corrected to Y, probe asks for the *current* value.
  Frequency-based decay actively fails this (the stale value got more mentions). Embedding's
  **stale rate is 0.73**: with recall ~1.0, its windows usually contain *both* values.

`interference`, `should-forget`, and `task-boundary` probe types are defined in the schema
but were not needed to reach the verdict.

### Anti-shiny-needle guard

A query-blind "retrieve the most salient turn" baseline (`SalienceOnly`) must **not** ace the
eval — if it does, the needles are trivially findable and we're measuring nothing. It is
asserted in CI (`test_salience_only_does_not_ace`); on plainly-stated needles it reaches 0.43
against a ceiling of 1.00.

## Tier 2: does context need repairing?

Three experiments, run on `claude-sonnet-4-6` and `claude-haiku-4-5`:

- **Kill-test (`run_killtest.py`)** — rebuild each supersession window with `live_only` /
  `stale_only` / `both` and ask the model. With both values present, drag = **0 / 15** on both
  tiers; with only the stale value, the model answers it 100% of the time, ~85% confidently.
  → suppression buys nothing; the harm is *absence*, and it's a window-completeness defect, not
  overconfidence.
- **Witnessless make-or-break + variance run (`variance_run.py`)** — the catastrophic case: a
  catering-menu draft with a life-threatening shellfish allergy aged out and the client's
  seafood preference still in the window. Measured as flag rates with 95% CIs (N=40, confirmed
  at N=120): no generic in-window marker reliably moves the spontaneous self-check rate.
- Supporting probes: `marker_reliability.py`, `prior_strength.py`, `band_probe.py`,
  `production_split.py`, `witnessless_unprobed.py`, `conditions.py`, `model.py`.

Tier-2 artifacts and the full reasoning trail live in `tier2_context_rot/` (`FINDINGS.md`,
`results/`).

## Run it

```bash
uv sync --extra dev

# Tier 1 — inner-loop frontier. Real models (OpenAI embeddings + Claude summarization),
# cached on disk.
uv run python scripts/run_frontier.py

# Deterministic, no network (CI / smoke). Hashing embeddings + extractive summaries.
ENGRAM_OFFLINE=1 uv run python scripts/run_frontier.py

# Tier 2 — supersession kill-test (is stale-but-present context harmful?)
uv run python tier2_context_rot/run_killtest.py \
  --models=claude-sonnet-4-6,claude-haiku-4-5-20251001

# Tier 2 — the rate-controlled variance run (the terminal result)
uv run python tier2_context_rot/variance_run.py 40

uv run pytest            # offline by default
```

Tier-1 outputs land in `results/`: `frontier_*.png`, `decay_sweep.png`, `frontier.json`,
`probe_results.csv`.

> **Offline mode is plumbing, not signal.** The hashing embeddings and extractive summarizer
> exist so the harness runs in CI without keys; they make summarization a strawman. The real
> verdict comes from the default (real-model) run.

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
scripts/
  run_frontier.py           Tier-1 frontier + decay sweep
  run_oracle_importance.py  oracle "perfect importance" ablation
tier2_context_rot/          Tier-2 end-to-end output study (see its README / FINDINGS.md)
data/scenarios/*.yaml       15 hand-authored scenarios (+ SCHEMA.md)
```

## Where the leverage actually is

Because suppression doesn't move the output and absence can't be flagged from inside the
window, the leverage in long-horizon agentic memory is **upstream**: a retention policy that
never evicts the never-reinforced critical fact (the stated-once allergy, the one-time vendor
blacklist). Liveness re-enters not as a filter and not as a confidence prompt, but as a
**retention priority** — the cost of dropping a fact should be a function of its consequence,
not its recency or frequency. That is the form of the original ACT-R "importance" bet the study
leaves standing, relocated from the ranker to the eviction policy, and still owed a
rate-controlled test.

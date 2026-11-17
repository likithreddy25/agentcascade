# AgentCascade (practice build)

![tests](https://github.com/likithreddy25/agentcascade/actions/workflows/tests.yml/badge.svg)

A cost-aware task router, hand-built from scratch as a learning exercise,
then extended with real Kafka streaming and a real paid LLM tier. This is
the "build it yourself" companion to `~/Downloads/agentcascade`, the
reference version — same architecture, independently implemented, then
pushed further with real GPT-4o-mini ground truth.

## The problem

Given a task, decide whether a cheap solver's answer can be trusted or
whether it needs to be escalated to an expensive one — without knowing
the right answer in advance. The domain here is arithmetic expressions
(`"(3 + 4) * (5 - 2)"`), chosen because it gives instant, free, 100%
certain ground truth (Python's own `eval`), which makes it possible to
build and debug the *routing logic* before ever touching a real model.

## Phase 1 — router + solver tiers (`tasks.py`, `tiers.py`, `features.py`, `train_router.py`)

Two deterministic solvers stand in for "cheap model" and "expensive
model": `NaiveTier` reads left-to-right with no operator precedence or
parentheses handling; `ExactTier` is always correct. A classifier is
trained on features computed from the expression text alone (length,
operator counts, nesting depth, division presence) to predict whether
`NaiveTier` will get a given expression right.

Real result: ROC-AUC 0.9733, F1 0.9404. Routing beats both
always-naive (71.75% accuracy) and always-exact (100% accuracy, but at
19200 cost units vs. 7216 for routing) on the accuracy/cost tradeoff —
96.25% accuracy at 62.4% less cost than always-exact.

## Phase 2 — caching (`cache.py`, `exact_cache.py`)

`FuzzySimilarityCache` (kept in the repo deliberately, not deleted) is a
worked example of a real bug: it treats lexically similar expressions as
cache hits, but arithmetic is an exact-answer domain — `"3 + 4 * (2 - 1)"`
and `"3 + 4 * (2 + 1)"` are one character apart and have completely
different answers (7 vs. 15). The fix, `ExactMatchCache`, only matches
after whitespace normalization — same expression, not "similar" ones.

## Phase 3 — Kafka (`transport.py`, `producer.py`, `worker.py`, `docker-compose.yml`)

`MessageProducer`/`MessageConsumer` are abstract interfaces with both a
real Kafka implementation and an in-memory one. `orchestrator.py`,
`evaluate.py`, and everything downstream depend only on the interface —
the same dependency-inversion pattern used again in Phase 4 to swap
solver tiers.

Real end-to-end benchmark (`evaluate.py`, 3600 tasks, 20% duplicate
injection): 96.97% accuracy, 69.3% cost savings vs. always-exact, 45.1%
cache hit rate.

## Phase 4 — real LLM tier (`openai_tier.py`, `anthropic_tier.py`, `train_router_real_llm.py`, `orchestrator_real_llm.py`)

`OpenAIModelTier` and `AnthropicModelTier` both implement `SolverTier` —
proof the interface from Phase 1 really is swappable across providers,
not just across deterministic stand-ins. Only `OpenAIModelTier` was
actually run, since this account only has an OpenAI key.

This phase asked a harder, more honest question than Phase 1: not "will
`NaiveTier` get this right" (a deterministic, structural failure mode)
but "will `gpt-4o-mini` actually get this right" (a real, occasionally
surprising failure mode) — measured by really calling the API on 1000
tasks and checking the answer.

**Real results** (`train_router_real_llm.py`, 1000 real gpt-4o-mini
calls, `$0.0064` total cost):

- gpt-4o-mini alone: correct on only **67–69%** of tasks. Real, measured
  arithmetic mistakes on multi-step expressions — not a hypothetical.
- Router trained on this real label: ROC-AUC **0.9874**, F1 **0.9574**.
- Feature importances shifted from Phase 1: `n_operators`/`n_numbers`
  dominate here (0.428 each), `max_nesting_depth` barely matters (0.0115)
  — a genuinely different failure mode than `NaiveTier`'s precedence
  blindness, which the router picked up on its own from real labels.
- Routing brings accuracy from 69% to **95.5%**.

**Cost refinement** (`evaluate_real_llm_cost.py`, no new API calls —
reuses the already-collected data): the naive "call the LLM, then maybe
escalate" design pays for every call even when it ends up discarding
~28% of them. Predicting trust *before* calling — using only the
expression text, no API call needed to do that — gets the identical
95.5% accuracy while skipping those calls outright: **28% less real
dollar spend**, implemented in `orchestrator_real_llm.py`.

## Real bugs found and fixed along the way

- `ModuleNotFoundError: No module named 'src'` — running scripts directly
  instead of as `python -m src.module_name` from the project root.
- Python/pip environment mismatch — `python3` resolved to Homebrew's
  Python even with a conda `(base)` env active; fixed with a dedicated
  venv.
- scikit-learn model/pickle version mismatch after a fresh install —
  fixed by re-training against the installed version and pinning it in
  `requirements.txt`.
- Kafka consumer-group join latency — a fresh consumer group's
  partition-assignment handshake took longer than the original idle-poll
  budget allowed; fixed by raising `idle_timeout`/`max_idle_polls`.
- `FuzzySimilarityCache` wrong-answer bug (see Phase 2) — found via a
  deliberate before/after demo, not by accident.

## Running it

```bash
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt

python -m src.tasks                        # sanity-check the generator
python -m src.train_router                 # Phase 1 router (no API key needed)
python -m src.cache                         # demonstrates the fuzzy-cache bug
python -m src.exact_cache                   # demonstrates the fix
python -m src.orchestrator                  # Phase 1+2 combined, deterministic tiers

docker compose up -d                        # Phase 3, requires Docker
python -m src.producer && python -m src.worker

export OPENAI_API_KEY=sk-...                # Phase 4, requires a real key
python -m src.train_router_real_llm         # real gpt-4o-mini calls, ~$0.01
python -m src.evaluate_real_llm_cost        # free — reuses collected data
python -m src.orchestrator_real_llm         # runtime predict-then-call orchestrator

python -m pytest tests/ -v                  # full test suite, no API calls
```

## What this is (and isn't)

This is a learning project built to understand cost-aware LLM routing,
tool-use interfaces, and streaming architecture hands-on — not a claim
that arithmetic is a domain where you'd actually want an LLM in the loop
at all (a calculator is strictly better for that). The real point,
demonstrated with real measured numbers rather than assumed ones: a
cheap classifier that knows when *not* to trust an LLM call can improve
both accuracy and cost at the same time, and the interfaces
(`SolverTier`, `MessageProducer`/`MessageConsumer`) that make this
swappable are the actual reusable idea.

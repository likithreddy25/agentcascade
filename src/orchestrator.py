"""
Step 7: the orchestrator. This is where Phase 1 (router) and Phase 2
(cache) actually become one system.

Decision order per task -- this order is deliberate, not arbitrary:
  1. Check the cache first. A cache hit is the cheapest possible outcome
     (zero solver cost), so it should always win if available.
  2. On a miss, run the naive tier (it's cheap -- always worth trying).
  3. Ask the trained router: should we trust naive's answer, or escalate?
  4. Whichever answer we end up trusting, store it in the cache so a
     future repeat of this exact task is free.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import joblib
import pandas as pd

from src.exact_cache import ExactMatchCache
from src.features import expression_to_features, FEATURE_COLUMNS
from src.tiers import NaiveTier, ExactTier, SolverTier

MODEL_PATH = Path(__file__).resolve().parent.parent / "data" / "router_model.joblib"


@dataclass
class Outcome:
    expression: str
    final_answer: float | None
    correct: bool | None
    source: str  # "cache" | "naive" | "naive_escalated_to_exact"
    cost_units: float


class Orchestrator:
    def __init__(self, escalation_threshold: float = 0.5,
                 cheap_tier: SolverTier | None = None,
                 expensive_tier: SolverTier | None = None):
        # Accepting tiers as constructor arguments -- rather than hardcoding
        # NaiveTier()/ExactTier() inside here -- is what makes swapping in
        # AnthropicModelTier later a zero-change operation for everything
        # in this class. This is the exact same idea as the
        # MessageProducer/MessageConsumer interface in transport.py:
        # depend on the interface (SolverTier), let the caller decide which
        # implementation to hand you.
        self.router_model = joblib.load(MODEL_PATH)
        self.cache = ExactMatchCache()
        self.naive = cheap_tier or NaiveTier()
        self.exact = expensive_tier or ExactTier()
        self.escalation_threshold = escalation_threshold
        # Running totals -- useful for the evaluation harness we'll build
        # next, so we don't have to recompute stats from scratch every time.
        self.n_tasks = 0
        self.n_cache_hits = 0
        self.n_escalations = 0
        self.total_cost_units = 0.0

    def _predict_trust_naive(self, expression: str) -> float:
        """Returns P(naive tier will get this right), per the trained
        router. This is exactly the model train_router.py produced --
        same feature columns, same order, loaded from disk instead of
        retrained."""
        features = expression_to_features(expression)
        row = pd.DataFrame([features], columns=FEATURE_COLUMNS)
        return float(self.router_model.predict_proba(row)[0, 1])

    def process(self, expression: str, true_answer: float | None = None) -> Outcome:
        self.n_tasks += 1

        cached = self.cache.lookup(expression)
        if cached is not None:
            self.n_cache_hits += 1
            correct = (abs(cached - true_answer) < 1e-6) if true_answer is not None else None
            return Outcome(expression, cached, correct, "cache", cost_units=0.0)

        naive_result = self.naive.solve_and_score(expression, true_answer)
        trust = self._predict_trust_naive(expression)

        if trust >= self.escalation_threshold and naive_result.success:
            final_answer = naive_result.answer
            cost = naive_result.cost_units
            source = "naive"
        else:
            exact_result = self.exact.solve_and_score(expression, true_answer)
            self.n_escalations += 1
            final_answer = exact_result.answer
            cost = naive_result.cost_units + exact_result.cost_units
            source = "naive_escalated_to_exact"

        self.total_cost_units += cost
        correct = (abs(final_answer - true_answer) < 1e-6) if (
            true_answer is not None and final_answer is not None) else None

        if final_answer is not None:
            self.cache.store(expression, final_answer)

        return Outcome(expression, final_answer, correct, source, cost)


if __name__ == "__main__":
    from src.tasks import generate_tasks

    orch = Orchestrator()
    tasks = generate_tasks(n=20, seed=999)
    for t in tasks:
        outcome = orch.process(t.expression, t.true_answer)
        print(f"{outcome.source:25s} correct={outcome.correct!s:5s} "
              f"cost={outcome.cost_units:5.1f}  {t.expression}")

    print(f"\nTotal cost for {orch.n_tasks} tasks: {orch.total_cost_units} units "
          f"({orch.n_escalations} escalations, {orch.n_cache_hits} cache hits)")

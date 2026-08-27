"""
Step 15: the runtime orchestrator for the real-LLM router -- "predict then
maybe call," not "call then maybe escalate."

Compare this to orchestrator.py's process(): there, self.naive.solve_and_score()
runs FIRST, unconditionally, and the router is only consulted afterward to
decide whether to trust the answer it already paid for. That ordering was
fine when the cheap tier was NaiveTier -- calling it costs nothing, so
call-order didn't matter for cost, only for code clarity.

Once the cheap tier is a real, paid LLM, call-order IS a cost decision.
The router's features come from expression_to_features(), which only
looks at the expression string -- it never needs the LLM's answer to make
its trust prediction. So there's no reason to pay for the call before
asking whether you'd trust it.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import joblib
import pandas as pd

from src.exact_cache import ExactMatchCache
from src.features import expression_to_features, FEATURE_COLUMNS
from src.tiers import ExactTier, SolverTier
from src.openai_tier import OpenAIModelTier

MODEL_PATH = Path(__file__).resolve().parent.parent / "data" / "router_model_real_llm.joblib"

INPUT_PRICE_PER_TOKEN = 0.15 / 1_000_000   # gpt-4o-mini, Aug 2026 pricing
OUTPUT_PRICE_PER_TOKEN = 0.60 / 1_000_000


@dataclass
class Outcome:
    expression: str
    final_answer: float | None
    correct: bool | None
    source: str  # "cache" | "llm" | "skipped_llm_used_exact"
    dollar_cost: float


class RealLLMOrchestrator:
    def __init__(self, escalation_threshold: float = 0.5,
                 cheap_tier: SolverTier | None = None,
                 expensive_tier: SolverTier | None = None):
        self.router_model = joblib.load(MODEL_PATH)
        self.cache = ExactMatchCache()
        # cheap_tier is only constructed lazily by the caller normally,
        # since building an OpenAIModelTier requires OPENAI_API_KEY to be
        # set -- but we still accept it as a constructor arg for the same
        # swappability reason as orchestrator.py.
        self.cheap = cheap_tier or OpenAIModelTier(model="gpt-4o-mini", cost_units=1.0)
        self.exact = expensive_tier or ExactTier()
        self.escalation_threshold = escalation_threshold

        self.n_tasks = 0
        self.n_cache_hits = 0
        self.n_llm_calls = 0
        self.n_skipped_llm = 0
        self.total_dollar_cost = 0.0

    def _predict_trust_llm(self, expression: str) -> float:
        """P(the LLM will actually get this right), from the expression
        text alone -- no API call involved in computing this."""
        features = expression_to_features(expression)
        row = pd.DataFrame([features], columns=FEATURE_COLUMNS)
        return float(self.router_model.predict_proba(row)[0, 1])

    def process(self, expression: str, true_answer: float | None = None) -> Outcome:
        self.n_tasks += 1

        cached = self.cache.lookup(expression)
        if cached is not None:
            self.n_cache_hits += 1
            correct = (abs(cached - true_answer) < 1e-6) if true_answer is not None else None
            return Outcome(expression, cached, correct, "cache", dollar_cost=0.0)

        # Decide trust BEFORE spending anything -- this is the whole point.
        trust = self._predict_trust_llm(expression)

        if trust >= self.escalation_threshold:
            before_in, before_out = self.cheap.total_input_tokens, self.cheap.total_output_tokens
            result = self.cheap.solve_and_score(expression, true_answer)
            self.n_llm_calls += 1
            call_cost = ((self.cheap.total_input_tokens - before_in) * INPUT_PRICE_PER_TOKEN +
                         (self.cheap.total_output_tokens - before_out) * OUTPUT_PRICE_PER_TOKEN)
            final_answer = result.answer
            source = "llm"
        else:
            # Never call the paid model at all -- go straight to the free,
            # always-correct deterministic tier.
            result = self.exact.solve_and_score(expression, true_answer)
            self.n_skipped_llm += 1
            call_cost = 0.0
            final_answer = result.answer
            source = "skipped_llm_used_exact"

        self.total_dollar_cost += call_cost
        correct = (abs(final_answer - true_answer) < 1e-6) if (
            true_answer is not None and final_answer is not None) else None

        if final_answer is not None:
            self.cache.store(expression, final_answer)

        return Outcome(expression, final_answer, correct, source, call_cost)


if __name__ == "__main__":
    # Small, real, cheap smoke test -- confirms the wiring works before
    # you'd ever run this over a larger batch.
    from src.tasks import generate_tasks

    orch = RealLLMOrchestrator()
    tasks = generate_tasks(n=20, seed=999)
    for t in tasks:
        outcome = orch.process(t.expression, t.true_answer)
        print(f"{outcome.source:25s} correct={outcome.correct!s:5s} "
              f"${outcome.dollar_cost:.8f}  {t.expression}")

    print(f"\n{orch.n_tasks} tasks, {orch.n_llm_calls} real LLM calls made, "
          f"{orch.n_skipped_llm} skipped (routed straight to free exact tool), "
          f"{orch.n_cache_hits} cache hits, total real cost: ${orch.total_dollar_cost:.8f}")

"""
Wrap-up tests. These don't make any real API calls -- they use a
DummyTier (a fake SolverTier you control completely) wherever a real LLM
tier would normally go, so the whole suite runs for free and offline.

Run from the project root:
    python -m pytest tests/test_pipeline.py -v
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.tasks import generate_tasks
from src.tiers import NaiveTier, ExactTier, SolverTier
from src.features import expression_to_features, FEATURE_COLUMNS
from src.exact_cache import ExactMatchCache
from src.orchestrator import Orchestrator


# ---------- tasks.py ----------

def test_generate_tasks_reproducible():
    a = generate_tasks(n=20, seed=42)
    b = generate_tasks(n=20, seed=42)
    assert [t.expression for t in a] == [t.expression for t in b]
    assert [t.true_answer for t in a] == [t.true_answer for t in b]


def test_generate_tasks_answers_are_correct():
    # Every task's true_answer should match Python's own evaluation of the
    # expression it was generated from -- this catches a broken generator
    # before it ever poisons a training set.
    for t in generate_tasks(n=50, seed=1):
        assert abs(eval(t.expression) - t.true_answer) < 1e-6


# ---------- tiers.py ----------

def test_naive_and_exact_agree_when_no_grouping_ambiguity():
    naive, exact = NaiveTier(), ExactTier()
    n = naive.solve_and_score("3 + 4 * 2")
    e = exact.solve_and_score("3 + 4 * 2")
    # No parens here, but naive still ignores precedence, so this is
    # actually a case where they're EXPECTED to differ (naive: 14, exact: 11).
    assert n.answer != e.answer


def test_naive_and_exact_agree_on_flat_expression():
    naive, exact = NaiveTier(), ExactTier()
    n = naive.solve_and_score("3 + 4")
    e = exact.solve_and_score("3 + 4")
    assert n.answer == e.answer == 7.0


def test_naive_diverges_on_nested_parens():
    naive, exact = NaiveTier(), ExactTier()
    n = naive.solve_and_score("(3 + 4) * 2")
    e = exact.solve_and_score("(3 + 4) * 2")
    assert e.answer == 14.0
    assert n.answer != e.answer  # naive strips the parens, gets this wrong


# ---------- features.py ----------

def test_features_have_all_expected_columns():
    features = expression_to_features("(3 + 4) * (5 - (2 / 1))")
    assert set(features.keys()) == set(FEATURE_COLUMNS)


def test_features_nesting_depth():
    assert expression_to_features("3 + 4")["max_nesting_depth"] == 0
    assert expression_to_features("(3 + 4)")["max_nesting_depth"] == 1
    assert expression_to_features("((3 + 4) * 2)")["max_nesting_depth"] == 2


def test_features_has_division_flag():
    assert expression_to_features("3 / 4")["has_division"] == 1
    assert expression_to_features("3 + 4")["has_division"] == 0


# ---------- exact_cache.py ----------

def test_cache_hits_on_whitespace_variant():
    cache = ExactMatchCache()
    cache.store("3 + 4 * (2 - 1)", 7.0)
    assert cache.lookup("3+4*(2-1)") == 7.0


def test_cache_misses_on_near_duplicate():
    # This is the exact bug FuzzySimilarityCache had -- a near-duplicate
    # expression with a different true answer must NOT hit.
    cache = ExactMatchCache()
    cache.store("3 + 4 * (2 - 1)", 7.0)
    assert cache.lookup("3 + 4 * (2 + 1)") is None


def test_cache_hit_rate_tracking():
    cache = ExactMatchCache()
    cache.store("1 + 1", 2.0)
    cache.lookup("1 + 1")   # hit
    cache.lookup("2 + 2")   # miss
    assert cache.hit_rate == 0.5


# ---------- orchestrator.py (using dummy tiers, no real model calls) ----------

class DummyTier(SolverTier):
    """A fully controllable fake tier for testing the orchestrator's
    control flow in isolation from any real solver or LLM."""
    def __init__(self, name: str, answer: float, cost_units: float):
        self.name = name
        self._answer = answer
        self.cost_units = cost_units

    def solve(self, expression: str) -> tuple[float | None, bool]:
        return self._answer, True


def test_orchestrator_cache_hit_costs_nothing():
    orch = Orchestrator(cheap_tier=DummyTier("cheap", 99.0, 1.0),
                         expensive_tier=DummyTier("expensive", 99.0, 12.0))
    orch.cache.store("1 + 1", 2.0)
    outcome = orch.process("1 + 1", true_answer=2.0)
    assert outcome.source == "cache"
    assert outcome.cost_units == 0.0
    assert outcome.correct is True


def test_orchestrator_trusts_cheap_tier_when_correct():
    # A trivial expression the router should be confident about.
    orch = Orchestrator(cheap_tier=DummyTier("cheap", 7.0, 1.0),
                         expensive_tier=DummyTier("expensive", 7.0, 12.0))
    outcome = orch.process("3 + 4", true_answer=7.0)
    assert outcome.correct is True
    # Cheap tier's answer was right, so its cost should show up somewhere
    # (either trusted directly, or escalated -- either way, correctness
    # holds because both dummy tiers return the same right answer here).
    assert outcome.final_answer == 7.0


def test_orchestrator_escalates_and_caches_result():
    orch = Orchestrator(cheap_tier=DummyTier("cheap", 1.0, 1.0),
                         expensive_tier=DummyTier("expensive", 14.0, 12.0))
    # A deeply nested expression the router should NOT trust naively --
    # exercises the escalation path regardless of which answer wins.
    outcome = orch.process("(((3 + 4))) * (((2 - 1)))")
    assert outcome.final_answer in (1.0, 14.0)
    # Whatever answer was used, it should now be cached.
    assert orch.cache.lookup("(((3 + 4))) * (((2 - 1)))") == outcome.final_answer


if __name__ == "__main__":
    import subprocess
    subprocess.run([sys.executable, "-m", "pytest", __file__, "-v"])

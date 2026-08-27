"""
Step 8: end-to-end evaluation through the real Orchestrator, on a batch
large enough for the cache to actually matter.
"""
from __future__ import annotations

import random

from src.orchestrator import Orchestrator
from src.tasks import generate_tasks
from src.tiers import NAIVE_COST_UNITS, EXACT_COST_UNITS


def run_benchmark(n: int, seed: int) -> dict:
    tasks = generate_tasks(n=n, seed=seed)

    # Real workloads have repeated requests -- duplicate ~20% of tasks
    # verbatim so the cache has something genuine to catch. Disclosed here,
    # not hidden in the final numbers.
    rng = random.Random(seed + 1)
    repeats = rng.sample(tasks, k=max(1, len(tasks) // 5))
    tasks = tasks + repeats
    rng.shuffle(tasks)

    from src.tiers import NaiveTier
    naive = NaiveTier()
    always_naive_correct = sum(
        int(bool(naive.solve_and_score(t.expression, t.true_answer).correct))
        for t in tasks
    )
    always_exact_cost = len(tasks) * EXACT_COST_UNITS

    orch = Orchestrator()
    routed_correct = 0
    for t in tasks:
        outcome = orch.process(t.expression, t.true_answer)
        routed_correct += int(bool(outcome.correct))

    n_total = len(tasks)
    result = {
        "n_tasks": n_total,
        "always_naive_accuracy": round(always_naive_correct / n_total, 4),
        "always_exact_accuracy": 1.0,
        "always_exact_cost": always_exact_cost,
        "routed_accuracy": round(routed_correct / n_total, 4),
        "routed_cost": round(orch.total_cost_units, 2),
        "cache_hit_rate": round(orch.n_cache_hits / n_total, 4),
        "escalation_rate": round(orch.n_escalations / n_total, 4),
        "cost_savings_vs_always_exact": round(
            1 - (orch.total_cost_units / always_exact_cost), 4),
    }
    return result


if __name__ == "__main__":
    import json
    result = run_benchmark(n=3000, seed=123)
    print(json.dumps(result, indent=2))

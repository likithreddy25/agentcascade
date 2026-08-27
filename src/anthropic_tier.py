"""
Step 12: a real SolverTier backed by an actual Claude model.

Honesty note worth internalizing: using an LLM to do arithmetic is a bit of
an artificial exercise -- a calculator is a better tool for that job than
a language model. That's fine here, because the POINT of this file isn't
"LLMs are the best way to add numbers." The point is proving that the
SolverTier interface really is swappable: everything in orchestrator.py,
evaluate.py, and worker.py was written against SolverTier, never against
NaiveTier/ExactTier specifically, so plugging in a real model here requires
touching NOTHING else. That's the actual architectural claim being tested.

Requires ANTHROPIC_API_KEY set in your own environment. Never hard-code a
key in this file or anywhere else in the repo.
"""
from __future__ import annotations

import os

from src.tiers import SolverTier


class AnthropicModelTier(SolverTier):
    def __init__(self, model: str, cost_units: float):
        import anthropic  # deferred import: optional dependency
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise RuntimeError(
                "ANTHROPIC_API_KEY is not set. Set it in your shell "
                "(export ANTHROPIC_API_KEY=sk-...) before using this tier."
            )
        self._client = anthropic.Anthropic(api_key=api_key)
        self._model = model
        self.name = model
        self.cost_units = cost_units
        # Real, actual token counts from the API -- not the made-up
        # cost_units constant. This is the number that would matter if you
        # were reporting a genuine cost/efficiency result.
        self.total_input_tokens = 0
        self.total_output_tokens = 0

    def solve(self, expression: str) -> tuple[float | None, bool]:
        response = self._client.messages.create(
            model=self._model,
            max_tokens=64,
            messages=[{
                "role": "user",
                "content": (
                    f"Evaluate this arithmetic expression and reply with "
                    f"ONLY the final numeric answer, nothing else: {expression}"
                ),
            }],
        )
        self.total_input_tokens += response.usage.input_tokens
        self.total_output_tokens += response.usage.output_tokens

        text = "".join(b.text for b in response.content if b.type == "text").strip()
        try:
            return float(text), True
        except ValueError:
            return None, False


if __name__ == "__main__":
    # Small, cheap smoke test -- only a handful of real API calls, not a
    # full evaluation run. Confirms the wiring works before you commit to
    # a bigger (and more expensive) run.
    from src.tasks import generate_tasks

    cheap = AnthropicModelTier(model="claude-haiku-4-5-20251001", cost_units=1.0)
    tasks = generate_tasks(n=5, seed=1)
    for t in tasks:
        result = cheap.solve_and_score(t.expression, t.true_answer)
        print(f"{t.expression:30s} true={t.true_answer:<10} "
              f"model_answer={result.answer} correct={result.correct}")

    print(f"\nReal tokens used: {cheap.total_input_tokens} in, "
          f"{cheap.total_output_tokens} out")

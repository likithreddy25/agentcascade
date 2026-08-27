"""
Step 12 (OpenAI version): a real SolverTier backed by OpenAI's API instead
of Anthropic's -- same SolverTier interface, different provider entirely.
If this drops in with zero changes anywhere else in the codebase, that's
the actual proof the abstraction from transport.py/tiers.py was worth
building, not just a nice diagram.

Requires OPENAI_API_KEY set in your own environment.
"""
from __future__ import annotations

import os

from src.tiers import SolverTier


class OpenAIModelTier(SolverTier):
    def __init__(self, model: str, cost_units: float):
        import openai  # deferred import: optional dependency
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError(
                "OPENAI_API_KEY is not set. Set it in your shell "
                "(export OPENAI_API_KEY=sk-...) before using this tier."
            )
        self._client = openai.OpenAI(api_key=api_key)
        self._model = model
        self.name = model
        self.cost_units = cost_units
        self.total_input_tokens = 0
        self.total_output_tokens = 0

    def solve(self, expression: str) -> tuple[float | None, bool]:
        response = self._client.chat.completions.create(
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
        self.total_input_tokens += response.usage.prompt_tokens
        self.total_output_tokens += response.usage.completion_tokens

        text = response.choices[0].message.content.strip()
        try:
            return float(text), True
        except ValueError:
            return None, False


if __name__ == "__main__":
    # Small, cheap smoke test first -- a handful of real calls, not a full run.
    from src.tasks import generate_tasks

    cheap = OpenAIModelTier(model="gpt-4o-mini", cost_units=1.0)
    tasks = generate_tasks(n=5, seed=1)
    for t in tasks:
        result = cheap.solve_and_score(t.expression, t.true_answer)
        print(f"{t.expression:30s} true={t.true_answer:<10} "
              f"model_answer={result.answer} correct={result.correct}")

    print(f"\nReal tokens used: {cheap.total_input_tokens} in, "
          f"{cheap.total_output_tokens} out")

"""
Step 14: prove the "predict before you pay" cost savings using data you
ALREADY collected -- zero new API calls, zero new spend.

Your train_router_real_llm.py run logged real usage: 33329 input tokens
and 2265 output tokens across 1000 real gpt-4o-mini calls. At OpenAI's
published rate for gpt-4o-mini -- $0.15 per 1M input tokens, $0.60 per 1M
output tokens (checked live, August 2026) -- that run cost about $0.0064
total. Tiny, because arithmetic prompts are short; the POINT isn't the
absolute dollar figure here, it's that the same percentage savings would
matter a lot more at real production volume or with a pricier model.

The comparison:
  - "always call the LLM": pay for gpt-4o-mini on every task, no routing.
  - "call-then-maybe-escalate" (what train_router_real_llm.py evaluated):
    pay for gpt-4o-mini on every task, THEN discard ~28% of those answers
    in favor of the free exact tool when the router doesn't trust them.
    Same accuracy lift, but you paid for calls you ended up throwing away.
  - "predict-then-maybe-call" (the fix): ask the router first, using only
    the expression text -- no API call needed for that. Only pay for
    gpt-4o-mini on the ~72% it's predicted to get right. Skip the call
    entirely on the rest, go straight to the free exact tool.

Accuracy is IDENTICAL between the last two -- same tasks end up routed to
the free tool either way. The only thing that changes is whether you paid
for a call you didn't end up trusting.
"""
from __future__ import annotations

import joblib
import pandas as pd

from src.features import FEATURE_COLUMNS

DATA_DIR_CSV = "data/labeled_tasks_real_llm.csv"
MODEL_PATH = "data/router_model_real_llm.joblib"

INPUT_PRICE_PER_TOKEN = 0.15 / 1_000_000   # gpt-4o-mini, Aug 2026 pricing
OUTPUT_PRICE_PER_TOKEN = 0.60 / 1_000_000

# From your actual run's printed line: "Real tokens used: 33329 in, 2265 out,
# across 1000 calls." Hardcoded here rather than re-derived, since the CSV
# doesn't store per-call token counts -- only the aggregate was printed.
TOTAL_CALLS_MADE = 1000
TOTAL_INPUT_TOKENS = 33329
TOTAL_OUTPUT_TOKENS = 2265
AVG_COST_PER_CALL = (TOTAL_INPUT_TOKENS / TOTAL_CALLS_MADE) * INPUT_PRICE_PER_TOKEN + \
                    (TOTAL_OUTPUT_TOKENS / TOTAL_CALLS_MADE) * OUTPUT_PRICE_PER_TOKEN


def main():
    df = pd.read_csv(DATA_DIR_CSV)
    model = joblib.load(MODEL_PATH)

    # Reproduce the exact same 80/20 split train_router_real_llm.py used --
    # no shuffling happened there, so this is deterministic and matches.
    split = int(len(df) * 0.8)
    test_df = df.iloc[split:].reset_index(drop=True)
    n_test = len(test_df)

    probs = model.predict_proba(test_df[FEATURE_COLUMNS])[:, 1]
    predicted_trust = probs >= 0.5
    gpt_mini_correct = test_df["gpt_mini_correct"].astype(bool).values

    # -- Always call the LLM, no routing at all --
    always_call_accuracy = gpt_mini_correct.mean()
    always_call_cost = n_test * AVG_COST_PER_CALL

    # -- Old design: call every task, escalate ~28% of them to the free tool --
    call_then_escalate_correct = gpt_mini_correct.copy()
    call_then_escalate_correct[~predicted_trust] = True  # free tool is always right
    call_then_escalate_accuracy = call_then_escalate_correct.mean()
    call_then_escalate_cost = n_test * AVG_COST_PER_CALL  # paid for ALL of them

    # -- New design: only call the LLM on tasks predicted trustworthy --
    predict_then_call_correct = gpt_mini_correct.copy()
    predict_then_call_correct[~predicted_trust] = True  # never called, free tool used
    predict_then_call_accuracy = predict_then_call_correct.mean()
    n_calls_made = int(predicted_trust.sum())
    predict_then_call_cost = n_calls_made * AVG_COST_PER_CALL

    print(f"Test set size: {n_test}")
    print(f"Average real cost per gpt-4o-mini call: ${AVG_COST_PER_CALL:.8f}\n")

    print(f"{'strategy':30s} {'accuracy':>10s} {'$ cost':>12s} {'llm calls made':>16s}")
    print(f"{'always call LLM':30s} {always_call_accuracy:>10.2%} "
          f"{always_call_cost:>12.6f} {n_test:>16d}")
    print(f"{'call then maybe escalate':30s} {call_then_escalate_accuracy:>10.2%} "
          f"{call_then_escalate_cost:>12.6f} {n_test:>16d}")
    print(f"{'predict then maybe call':30s} {predict_then_call_accuracy:>10.2%} "
          f"{predict_then_call_cost:>12.6f} {n_calls_made:>16d}")

    savings_pct = 1 - (predict_then_call_cost / call_then_escalate_cost)
    print(f"\nSame accuracy as 'call then escalate' ({call_then_escalate_accuracy:.2%} "
          f"vs {predict_then_call_accuracy:.2%}), for {savings_pct:.1%} less real "
          f"dollar spend on the LLM -- by simply not calling it on tasks the "
          f"router already knows it'll get wrong.")


if __name__ == "__main__":
    main()

"""
Step 13: retrain the router on REAL ground truth from GPT-4o-mini instead
of the deterministic NaiveTier. Same features, same training procedure --
only the label changes: instead of "did the naive left-to-right evaluator
get this right," it's "did gpt-4o-mini actually get this right," measured
by really calling the API and checking the answer.

Cost estimate before you run this: your 5-call smoke test used ~185 input
+ 10 output tokens. At gpt-4o-mini's pricing that's a small fraction of a
cent per call -- 1000 calls should cost well under $0.10 total. Still,
this makes real API calls that cost real (if tiny) money, on your account,
with your key -- there's no way to run this without that being true.

This writes the labeled dataset to a CSV as it goes, specifically so that
if something crashes partway through, you don't lose the API spend you
already made -- you can resume from what's saved rather than starting over.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import (
    roc_auc_score, precision_score, recall_score, f1_score, accuracy_score,
)

from src.tasks import generate_tasks
from src.features import expression_to_features, FEATURE_COLUMNS
from src.openai_tier import OpenAIModelTier

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
LABELED_CSV = DATA_DIR / "labeled_tasks_real_llm.csv"


def build_labeled_dataset(n: int, seed: int) -> pd.DataFrame:
    DATA_DIR.mkdir(exist_ok=True)
    tasks = generate_tasks(n=n, seed=seed)
    cheap = OpenAIModelTier(model="gpt-4o-mini", cost_units=1.0)

    rows = []
    with open(LABELED_CSV, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([*FEATURE_COLUMNS, "expression", "true_answer",
                          "gpt_mini_correct", "gpt_mini_answer"])

        for i, t in enumerate(tasks):
            result = cheap.solve_and_score(t.expression, t.true_answer)
            features = expression_to_features(t.expression)
            row = {
                **features,
                "expression": t.expression,
                "true_answer": t.true_answer,
                "gpt_mini_correct": int(bool(result.correct)),
                "gpt_mini_answer": result.answer,
            }
            rows.append(row)
            writer.writerow([row[c] for c in FEATURE_COLUMNS] +
                             [row["expression"], row["true_answer"],
                              row["gpt_mini_correct"], row["gpt_mini_answer"]])
            f.flush()  # write to disk immediately, don't lose progress on a crash

            if (i + 1) % 50 == 0:
                print(f"  {i + 1}/{n} tasks called...")

    print(f"\nReal tokens used: {cheap.total_input_tokens} in, "
          f"{cheap.total_output_tokens} out, across {n} calls")
    return pd.DataFrame(rows)


def main(n: int = 1000, seed: int = 7):
    if LABELED_CSV.exists():
        print(f"Found existing {LABELED_CSV} -- loading it instead of "
              f"making new API calls. Delete the file if you want to "
              f"regenerate it from scratch.")
        df = pd.read_csv(LABELED_CSV)
    else:
        print(f"Calling gpt-4o-mini on {n} real tasks (this costs a small "
              f"amount of real money on your OpenAI account)...")
        df = build_labeled_dataset(n=n, seed=seed)

    gpt_mini_baseline_acc = df["gpt_mini_correct"].mean()
    print(f"\ngpt-4o-mini alone is correct on {gpt_mini_baseline_acc:.2%} "
          f"of {len(df)} real tasks -- this is the 'always trust the LLM' "
          f"baseline you're trying to beat.")

    split = int(len(df) * 0.8)
    train_df, test_df = df.iloc[:split], df.iloc[split:]
    X_train, y_train = train_df[FEATURE_COLUMNS], train_df["gpt_mini_correct"]
    X_test, y_test = test_df[FEATURE_COLUMNS], test_df["gpt_mini_correct"]

    model = GradientBoostingClassifier(n_estimators=150, max_depth=3,
                                        learning_rate=0.1, random_state=42)
    model.fit(X_train, y_train)
    probs = model.predict_proba(X_test)[:, 1]
    preds = (probs >= 0.5).astype(int)

    metrics = {
        "roc_auc": round(roc_auc_score(y_test, probs), 4),
        "accuracy": round(accuracy_score(y_test, preds), 4),
        "precision": round(precision_score(y_test, preds), 4),
        "recall": round(recall_score(y_test, preds), 4),
        "f1": round(f1_score(y_test, preds), 4),
    }
    print("\nRouter metrics (predicting 'will gpt-4o-mini actually get this right'):")
    print(json.dumps(metrics, indent=2))

    importances = dict(zip(FEATURE_COLUMNS, model.feature_importances_.round(4)))
    print("\nFeature importances:")
    for name, val in sorted(importances.items(), key=lambda kv: -kv[1]):
        print(f"  {name:25s} {val}")

    # Cost story: escalating goes to the FREE, always-correct deterministic
    # ExactTier -- not a second paid LLM call. So routing doesn't trade
    # accuracy for cost the way the naive/exact version did; it can raise
    # accuracy toward 100% while REDUCING total dollar spend, since every
    # escalated task costs nothing further. That's a genuinely different
    # (and arguably more realistic) story than Phase 1's version: this is
    # the real argument for giving an LLM a calculator tool instead of
    # trusting its raw arithmetic.
    n_test = len(test_df)
    gpt_mini_correct = test_df["gpt_mini_correct"].values
    escalate = (preds == 0)
    routed_correct = gpt_mini_correct.copy()
    routed_correct[escalate] = 1  # ExactTier is always right once escalated
    routed_acc = routed_correct.mean()

    print(f"\nalways-trust-gpt-mini accuracy: {gpt_mini_correct.mean():.2%}")
    print(f"routed accuracy (escalate to free exact tool when unsure): {routed_acc:.2%}")
    print(f"escalation rate: {escalate.mean():.1%} "
          f"(fraction of paid LLM calls whose result gets discarded and "
          f"replaced by the free tool -- meaning you paid for a call you "
          f"didn't end up needing; see the note about calling the router "
          f"BEFORE the LLM instead, to avoid paying for those at all)")

    import joblib
    joblib.dump(model, DATA_DIR / "router_model_real_llm.joblib")
    print(f"\nSaved -> {DATA_DIR / 'router_model_real_llm.joblib'}")


if __name__ == "__main__":
    main(n=1000, seed=7)

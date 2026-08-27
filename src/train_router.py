"""
Step 4: build a labeled dataset, train a classifier, evaluate it honestly.

The core idea of "labeling": we do NOT hand-decide which expressions are
hard. We generate a batch of expressions, run NaiveTier and ExactTier on
each one for real, and the label is simply: did naive's answer match
exact's answer? That's it -- the label is a measured fact, not a guess.
"""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import (
    roc_auc_score, precision_score, recall_score, f1_score, accuracy_score,
)

from src.tasks import generate_tasks
from src.features import expression_to_features, FEATURE_COLUMNS
from src.tiers import NaiveTier, ExactTier, NAIVE_COST_UNITS, EXACT_COST_UNITS


def build_labeled_dataset(n: int, seed: int) -> pd.DataFrame:
    tasks = generate_tasks(n=n, seed=seed)
    naive, exact = NaiveTier(), ExactTier()
    rows = []
    for t in tasks:
        naive_result = naive.solve_and_score(t.expression, true_answer=t.true_answer)
        # Sanity check while you're building this: ExactTier should NEVER
        # disagree with the generator's own ground truth (both use the same
        # eval() logic). If this assertion ever fires, the bug is upstream
        # of the label -- go check tasks.py or tiers.py, not the router.
        exact_result = exact.solve_and_score(t.expression, true_answer=t.true_answer)
        assert exact_result.correct, f"ExactTier disagreed on {t.expression}!"

        features = expression_to_features(t.expression)
        rows.append({
            **features,
            "expression": t.expression,
            "naive_correct": int(bool(naive_result.correct)),
        })
    return pd.DataFrame(rows)


def main():
    print("Generating labeled dataset...")
    df = build_labeled_dataset(n=8000, seed=42)
    naive_baseline_acc = df["naive_correct"].mean()
    print(f"Naive tier alone is correct on {naive_baseline_acc:.2%} of {len(df)} tasks "
          f"-- this is the 'always-naive' baseline you're trying to beat.")

    # 80/20 split. Since these tasks are independent and randomly generated
    # (no shared history the way fraud transactions on the SAME account
    # would have), a random split is fine here -- no leakage risk the way
    # there would be if, say, multiple near-identical expressions from the
    # same "session" could land on both sides of the split.
    split = int(len(df) * 0.8)
    train_df, test_df = df.iloc[:split], df.iloc[split:]
    X_train, y_train = train_df[FEATURE_COLUMNS], train_df["naive_correct"]
    X_test, y_test = test_df[FEATURE_COLUMNS], test_df["naive_correct"]

    model = GradientBoostingClassifier(n_estimators=150, max_depth=3,
                                        learning_rate=0.1, random_state=42)
    model.fit(X_train, y_train)

    probs = model.predict_proba(X_test)[:, 1]  # P(naive will succeed)
    preds = (probs >= 0.5).astype(int)

    metrics = {
        "roc_auc": round(roc_auc_score(y_test, probs), 4),
        "accuracy": round(accuracy_score(y_test, preds), 4),
        "precision": round(precision_score(y_test, preds), 4),
        "recall": round(recall_score(y_test, preds), 4),
        "f1": round(f1_score(y_test, preds), 4),
    }
    print("\nRouter classification metrics:")
    print(json.dumps(metrics, indent=2))

    # Feature importances -- check these against the hypotheses written in
    # features.py's comments. Do they actually agree?
    importances = dict(zip(FEATURE_COLUMNS, model.feature_importances_.round(4)))
    print("\nFeature importances (which features did the model actually rely on?):")
    for name, val in sorted(importances.items(), key=lambda kv: -kv[1]):
        print(f"  {name:25s} {val}")

    # ---- the actual point: does routing beat both baselines? ----
    n = len(test_df)
    naive_correct = test_df["naive_correct"].values

    always_naive_acc = naive_correct.mean()
    always_naive_cost = n * NAIVE_COST_UNITS

    always_exact_acc = 1.0
    always_exact_cost = n * EXACT_COST_UNITS

    # Routed: always pay naive's cost; ALSO pay exact's cost whenever the
    # model predicts naive will fail (we escalate). After escalating,
    # correctness is 1 for that task, since exact is always right.
    escalate = (preds == 0)
    routed_correct = naive_correct.copy()
    routed_correct[escalate] = 1
    routed_acc = routed_correct.mean()
    routed_cost = n * NAIVE_COST_UNITS + escalate.sum() * EXACT_COST_UNITS

    print("\nCost/accuracy comparison on the held-out test set:")
    print(f"  always-naive : {always_naive_acc:.2%} accuracy, {always_naive_cost:.0f} cost units")
    print(f"  always-exact : {always_exact_acc:.2%} accuracy, {always_exact_cost:.0f} cost units")
    print(f"  routed       : {routed_acc:.2%} accuracy, {routed_cost:.0f} cost units "
          f"(escalated {escalate.mean():.1%} of tasks)")

    savings = 1 - (routed_cost / always_exact_cost)
    print(f"\n  -> routing gets {routed_acc:.2%} accuracy at {savings:.1%} lower cost "
          f"than always using the expensive tier.")

    # Persist the trained model to disk. Why this matters: training just
    # now took real (if short) time and real data. Without saving it, every
    # single script that wants to USE the router -- the orchestrator, an
    # API server, anything -- would have to retrain from scratch every time
    # it starts up. joblib is the standard way to serialize scikit-learn
    # models specifically (plain pickle also works, but joblib is more
    # efficient for the numpy arrays inside sklearn models).
    data_dir = Path(__file__).resolve().parent.parent / "data"
    data_dir.mkdir(exist_ok=True)
    model_path = data_dir / "router_model.joblib"
    joblib.dump(model, model_path)
    print(f"\nSaved trained model -> {model_path}")


if __name__ == "__main__":
    main()

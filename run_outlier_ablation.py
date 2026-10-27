"""
Compares every dataset with and without the outlier rule (R8).
Usage:
    python run_outlier_ablation.py

For each dataset the profile and plan are built as usual, then each of the
4 models is scored twice with the same CV folds: once with outlier clipping
and once with it switched off. The table is printed and saved to
results/outlier_ablation.csv. To add a dataset, add a line to DATASETS.
"""
import os
import warnings

import pandas as pd
from sklearn.datasets import load_breast_cancer

from src.data_loader import load_telco_churn
from src.profiler import DataProfiler
from src.preprocessor import PreprocessingDecider
from src.model_selector import ModelSelector
from src.ablation import run_ablation

MAX_ROWS = 10000   # larger datasets are sampled, same as the app's fast mode
OUTPUT_PATH = "results/outlier_ablation.csv"

# (name, function that loads the DataFrame, target column)
DATASETS = [
    ("adult", lambda: pd.read_csv("adult_income.csv"), "class"),
    ("telco", load_telco_churn, "Churn"),
    ("mushroom", lambda: pd.read_csv("mushroom.csv"), "target"),
    ("breast_cancer", lambda: load_breast_cancer(as_frame=True).frame, "target"),
]


def compare_dataset(name, df, target):
    if len(df) > MAX_ROWS:
        df = df.sample(n=MAX_ROWS, random_state=42)

    profile = DataProfiler(df, target_column=target).profile()
    plan = PreprocessingDecider(profile).decide()
    result = ModelSelector(df, profile, plan).select()
    n_clipped = sum(d.outlier_action == "clip" for d in plan.column_decisions)
    n_numeric = sum(d.scaling_action == "standard_scale" for d in plan.column_decisions)
    print(f"{name}: {len(df)} rows, metric={result.metric_used}, "
          f"{n_clipped} of {n_numeric} numeric columns clipped")

    rows = []
    for candidate in result.candidate_results:
        scores = run_ablation(df, profile, plan, candidate.model_name, result.metric_used,
                              cv_folds=result.cv_folds_used, variant_names=["no_outlier_clipping_R8"])
        with_r8, without_r8 = scores[0].score, scores[1].score

        # the "with" score should be the same number the model selector reported
        selector_score = getattr(candidate, result.metric_used)
        if abs(with_r8 - selector_score) > 1e-9:
            print(f"  WARNING: {candidate.model_name} ablation score {with_r8} != selector score {selector_score}")

        rows.append({
            "dataset": name, "n_rows": len(df), "metric": result.metric_used,
            "model": candidate.model_name, "columns_clipped": n_clipped,
            "without_R8": round(without_r8, 4), "with_R8": round(with_r8, 4),
            "difference": round(with_r8 - without_r8, 4),   # positive = clipping helped
        })

    # best model with R8 vs best model without R8 (they can be different models)
    best_without = max(r["without_R8"] for r in rows)
    best_with = max(r["with_R8"] for r in rows)
    rows.append({
        "dataset": name, "n_rows": len(df), "metric": result.metric_used,
        "model": "BEST OF 4 MODELS", "columns_clipped": n_clipped,
        "without_R8": best_without, "with_R8": best_with,
        "difference": round(best_with - best_without, 4),
    })
    return rows


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    all_rows = []
    for name, loader, target in DATASETS:
        all_rows.extend(compare_dataset(name, loader(), target))

    table = pd.DataFrame(all_rows)
    os.makedirs("results", exist_ok=True)
    table.to_csv(OUTPUT_PATH, index=False)

    print("\n=== OUTLIER RULE (R8): WITHOUT vs WITH ===")
    print(table.to_string(index=False))
    print(f"\nSaved to {OUTPUT_PATH}")

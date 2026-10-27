"""
Repeated with/without comparison of the outlier rule (R8).
Usage:
    python run_outlier_comparison.py

Every model is scored with and without R8 on 5 different cross-validation
seeds (the CV shuffle changes, the model seeds stay fixed). Mean and standard
deviation over the seeds are reported per model, plus a plain verdict:
  - "within 1 std": the difference is no bigger than the spread of the
    without-R8 scores across seeds, so it could just be CV noise
  - "better" / "worse": the difference is bigger than that spread
This is NOT a significance test, it is only a rough size check.
Results are saved to results/outlier_repeated.csv.
"""
import os
import warnings

import numpy as np
import pandas as pd
from sklearn.datasets import load_diabetes, load_wine

from src.profiler import DataProfiler
from src.target_preparer import prepare_target
from src.preprocessor import PreprocessingDecider
from src.model_selector import _build_candidates
from src.ablation import run_ablation

SEEDS = [0, 1, 2, 3, 4]
ADULT_ROWS = None   # None = the full 48,842 rows (about 10 minutes for all seeds); set e.g. 20000 to go faster
OUTPUT_PATH = "results/outlier_repeated.csv"


def choose_metric(profile) -> str:
    """Same rule the model selector uses: F1 for imbalanced or multiclass data, else accuracy."""
    return "f1" if (profile.is_imbalanced or profile.n_classes > 2) else "accuracy"


def cv_folds_for(df, target, max_folds=5) -> int:
    """Same rule the model selector uses: never more folds than the smallest class has rows."""
    return min(max_folds, int(df[target].value_counts().min()))


def load_diabetes_3class():
    """Diabetes progression split into 3 equal-sized groups, using the app's own target preparation."""
    df, _notes = prepare_target(load_diabetes(as_frame=True).frame, "target", n_bins=3)
    return df


def load_wine_df():
    return load_wine(as_frame=True).frame


def score_with_and_without(df, target, seeds=SEEDS):
    """
    Scores every model with and without R8 for every CV seed.
    Returns (rows, columns_clipped, metric); one row per model and seed.
    """
    profile = DataProfiler(df, target_column=target).profile()
    plan = PreprocessingDecider(profile).decide()
    metric = choose_metric(profile)
    folds = cv_folds_for(df, target)
    n_clipped = sum(d.outlier_action == "clip" for d in plan.column_decisions)

    rows = []
    for model_name in _build_candidates():
        for seed in seeds:
            scores = run_ablation(df, profile, plan, model_name, metric, cv_folds=folds,
                                  variant_names=["no_outlier_clipping_R8"], random_state=seed)
            rows.append({"model": model_name, "seed": seed,
                         "with_R8": scores[0].score, "without_R8": scores[1].score})
    return rows, n_clipped, metric


def verdict(with_scores, without_scores) -> str:
    """Compares the mean difference to one std of the without-R8 scores (over seeds)."""
    diff = float(np.mean(with_scores) - np.mean(without_scores))
    if diff == 0:
        return "identical"
    if abs(diff) <= np.std(without_scores, ddof=1):
        return "within 1 std"
    return "better" if diff > 0 else "worse"


def summarize(rows, n_clipped, metric):
    """One summary row per model: mean and std over the seeds with and without R8."""
    table = pd.DataFrame(rows)
    summary = []
    for model_name, group in table.groupby("model", sort=False):
        with_scores = group["with_R8"].to_numpy()
        without_scores = group["without_R8"].to_numpy()
        summary.append({
            "model": model_name,
            "metric": metric,
            "columns_clipped": n_clipped,
            "mean_without": round(float(without_scores.mean()), 4),
            "std_without": round(float(without_scores.std(ddof=1)), 4),
            "mean_with": round(float(with_scores.mean()), 4),
            "std_with": round(float(with_scores.std(ddof=1)), 4),
            "difference": round(float(with_scores.mean() - without_scores.mean()), 4),
            "seeds_with_better": int((with_scores > without_scores).sum()),
            "n_seeds": len(group),
            "verdict": verdict(with_scores, without_scores),
        })
    return summary


def compare_dataset(name, df, target, seeds=SEEDS):
    rows, n_clipped, metric = score_with_and_without(df, target, seeds)
    return [{"dataset": name, "n_rows": len(df), **r} for r in summarize(rows, n_clipped, metric)]


if __name__ == "__main__":
    warnings.filterwarnings("ignore")

    adult = pd.read_csv("adult_income.csv")
    if ADULT_ROWS is not None and len(adult) > ADULT_ROWS:
        adult = adult.sample(n=ADULT_ROWS, random_state=42)

    datasets = [("adult", adult, "class"), ("wine", load_wine_df(), "target"),
                ("diabetes_3class", load_diabetes_3class(), "target")]

    all_rows = []
    for name, df, target in datasets:
        print(f"running {name} ({len(df)} rows) ...", flush=True)
        all_rows.extend(compare_dataset(name, df, target))

    table = pd.DataFrame(all_rows)
    os.makedirs("results", exist_ok=True)
    table.to_csv(OUTPUT_PATH, index=False)
    print("\n=== R8 WITHOUT vs WITH, mean (std) over 5 CV seeds ===")
    print(table.to_string(index=False))
    print(f"\nSaved to {OUTPUT_PATH}")

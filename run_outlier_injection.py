"""
Controlled outlier injection experiment.
Usage:
    python run_outlier_injection.py

Breast cancer and Wine are clean datasets, so R8 has little to do on them.
Here we add known outliers on purpose: in the 3 most important columns a
fraction of the values is replaced by mean + or - 10*std (sign chosen at
random). Then the full pipeline is scored with and without R8, over 5 CV
seeds, for injection rates of 0%, 2%, 5% and 10%.

The injection is done on the whole dataframe before cross-validation, with a
fixed seed (INJECT_SEED), so every run sees exactly the same corrupted data.
Results are saved to results/outlier_injection.csv.
"""
import os
import warnings

import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer, load_wine
from sklearn.ensemble import RandomForestClassifier

from run_outlier_comparison import score_with_and_without, summarize

RATES = [0.0, 0.02, 0.05, 0.10]
INJECT_SEED = 0   # fixed, so the injected rows and signs are the same every run
OUTPUT_PATH = "results/outlier_injection.csv"


def top_columns(df, target, k=3):
    """The k most important feature columns according to a quick random forest."""
    X = df.drop(columns=[target])
    forest = RandomForestClassifier(n_estimators=100, random_state=0, n_jobs=-1)
    forest.fit(X, df[target])
    order = np.argsort(forest.feature_importances_)[::-1][:k]
    return [X.columns[i] for i in order]


def inject_outliers(df, columns, rate, seed=INJECT_SEED):
    """
    Returns (new_df, positions). In each given column round(rate * n_rows) values
    are replaced by mean + or - 10*std of the original column. positions maps
    column -> row positions that were changed. The input df is not modified.
    """
    rng = np.random.default_rng(seed)
    new_df = df.copy()
    positions = {}
    n_extreme = int(round(rate * len(df)))
    for col in columns:
        original = df[col].astype(float)
        mean, std = original.mean(), original.std()
        rows = rng.choice(len(df), size=n_extreme, replace=False)
        signs = rng.choice([-1.0, 1.0], size=n_extreme)
        values = new_df[col].astype(float).to_numpy()
        values[rows] = mean + signs * 10 * std
        new_df[col] = values
        positions[col] = rows
    return new_df, positions


def run_injection(name, df, target, rates=RATES, seeds=None):
    columns = top_columns(df, target)
    print(f"{name}: injecting into {columns}", flush=True)
    results = []
    for rate in rates:
        injected_df, _ = inject_outliers(df, columns, rate)
        kwargs = {} if seeds is None else {"seeds": seeds}
        rows, n_clipped, metric = score_with_and_without(injected_df, target, **kwargs)
        for r in summarize(rows, n_clipped, metric):
            results.append({"dataset": name, "injection_rate_pct": int(round(rate * 100)),
                            "injected_columns": ", ".join(columns), **r})
    return results


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    datasets = [("breast_cancer", load_breast_cancer(as_frame=True).frame, "target"),
                ("wine", load_wine(as_frame=True).frame, "target")]

    all_rows = []
    for name, df, target in datasets:
        all_rows.extend(run_injection(name, df, target))

    table = pd.DataFrame(all_rows)
    os.makedirs("results", exist_ok=True)
    table.to_csv(OUTPUT_PATH, index=False)

    show = ["dataset", "injection_rate_pct", "model", "metric", "columns_clipped",
            "mean_without", "std_without", "mean_with", "std_with", "difference", "verdict"]
    print("\n=== INJECTED OUTLIERS: R8 WITHOUT vs WITH, mean/std over 5 CV seeds ===")
    print(table[show].to_string(index=False))
    print(f"\nSaved to {OUTPUT_PATH}")

"""
Quick manual test — confirms Modules 1, 2, and 3 work end to end.
Loads a dataset, profiles it, decides preprocessing, then selects a model
and compares it against a fixed baseline.
"""
import json
import pandas as pd
from sklearn.datasets import load_breast_cancer
from src.profiler import DataProfiler
from src.preprocessor import PreprocessingDecider
from src.model_selector import ModelSelector


def load_sample_dataset() -> pd.DataFrame:
    data = load_breast_cancer(as_frame=True)
    return data.frame


if __name__ == "__main__":
    df = load_sample_dataset()

    profile = DataProfiler(df, target_column="target").profile()
    print("=== PROFILE ===")
    print(f"{profile.n_rows} rows, {profile.n_cols} cols, imbalanced={profile.is_imbalanced}")

    plan = PreprocessingDecider(profile).decide()
    print("\n=== PREPROCESSING PLAN SUMMARY ===")
    for line in plan.summary:
        print("-", line)

    print("\n=== MODEL SELECTION (this will take ~10-20 seconds) ===")
    result = ModelSelector(df, profile, plan).select()
    print(json.dumps(result.to_dict(), indent=2))
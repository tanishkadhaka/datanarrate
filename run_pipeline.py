"""
CLI pipeline runner. Usage:
    python run_pipeline.py <path_to_csv> <target_column>

Prints the full pipeline output to the console — profile, preprocessing
steps, model comparison, confusion matrices, and narration — without
touching Streamlit. Useful for quickly checking any dataset.
"""
import sys
import json
import pandas as pd

from src.profiler import DataProfiler
from src.preprocessor import PreprocessingDecider
from src.model_selector import ModelSelector, InsufficientDataError
from src.narrator import Narrator
from src.storage import RunStorage


def main():
    if len(sys.argv) != 3:
        print("Usage: python run_pipeline.py <path_to_csv> <target_column>")
        sys.exit(1)

    csv_path, target_column = sys.argv[1], sys.argv[2]
    df = pd.read_csv(csv_path)
    print(f"Loaded {df.shape[0]} rows, {df.shape[1]} columns from {csv_path}")
    print(f"Target column: '{target_column}'")
    print(f"Target value counts:\n{df[target_column].value_counts()}\n")

    if target_column not in df.columns:
        print(f"ERROR: '{target_column}' not found. Available columns: {df.columns.tolist()}")
        sys.exit(1)

    profile = DataProfiler(df, target_column=target_column).profile()
    print(f"\n=== PROFILE ===")
    print(f"{profile.n_rows} rows, {profile.n_cols} cols, "
          f"{profile.n_classes} classes, imbalanced={profile.is_imbalanced}")

    plan = PreprocessingDecider(profile).decide()
    print(f"\n=== PREPROCESSING STEP TRACE ===")
    for step in plan.step_trace:
        print("-", step)

    try:
        print(f"\n=== MODEL SELECTION (running 4 models, this may take a while) ===")
        result = ModelSelector(df, profile, plan).select()
    except InsufficientDataError as e:
        print(f"ERROR: {e}")
        sys.exit(1)

    if result.cv_folds_used < 5:
        print(f"NOTE: used {result.cv_folds_used}-fold CV instead of 5 (smallest class is small)")

    print(json.dumps(result.to_dict(), indent=2))

    print(f"\n=== FINAL JUSTIFICATION ===")
    print(result.final_justification)

    print(f"\n=== NARRATION ===")
    narration = Narrator().narrate(profile, plan, result)
    print(narration)

    storage = RunStorage()
    run_id = storage.save_run(csv_path, profile, plan, result, narration)
    print(f"\nSaved as run #{run_id}")


if __name__ == "__main__":
    main()
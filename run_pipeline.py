"""
CLI pipeline runner. Usage:
    python run_pipeline.py <path_to_csv> <target_column>

Prepares the target column (binning, rare-class merging, leakage check),
then prints the full pipeline output to the console. Runs on the full
dataset, so large files take a while.
"""
import sys
import json
import pandas as pd

from src.profiler import DataProfiler
from src.preprocessor import PreprocessingDecider
from src.model_selector import ModelSelector, InsufficientDataError
from src.narrator import Narrator
from src.storage import RunStorage
from src.target_preparer import prepare_target, InvalidTargetError


def main():
    if len(sys.argv) != 3:
        print("Usage: python run_pipeline.py <path_to_csv> <target_column>")
        sys.exit(1)

    csv_path, target_column = sys.argv[1], sys.argv[2]

    df = pd.read_csv(csv_path)
    print(f"Loaded {df.shape[0]} rows, {df.shape[1]} columns from {csv_path}")

    if target_column not in df.columns:
        print(f"ERROR: '{target_column}' not found. Available columns: {df.columns.tolist()}")
        sys.exit(1)

    print(f"Target column: '{target_column}'")
    print(f"Raw target value counts (top 15):\n{df[target_column].value_counts().head(15)}\n")

    try:
        df, target_notes = prepare_target(df, target_column)
    except InvalidTargetError as e:
        print(f"ERROR: can't use '{target_column}' as the target: {e}")
        sys.exit(1)

    print("=== TARGET PREPARATION ===")
    for note in target_notes:
        print("-", note)

    profile = DataProfiler(df, target_column=target_column).profile()
    print("\n=== PROFILE ===")
    print(f"{profile.n_rows} rows, {profile.n_cols} cols, "
          f"{profile.n_classes} classes, imbalanced={profile.is_imbalanced}")

    plan = PreprocessingDecider(profile).decide()
    plan.summary = [f"Target preparation: {n}" for n in target_notes] + plan.summary

    print("\n=== PREPROCESSING STEP TRACE ===")
    for step in plan.step_trace:
        print("-", step)

    try:
        print("\n=== MODEL SELECTION (running 4 models, this may take a while) ===")
        result = ModelSelector(df, profile, plan).select()
    except InsufficientDataError as e:
        print(f"ERROR: {e}")
        sys.exit(1)

    if result.cv_folds_used < 5:
        print(f"NOTE: used {result.cv_folds_used}-fold CV instead of 5 (smallest class is small)")

    print(json.dumps(result.to_dict(), indent=2))

    print("\n=== FINAL JUSTIFICATION ===")
    print(result.final_justification)

    print("\n=== NARRATION ===")
    narration = Narrator().narrate(profile, plan, result)
    print(narration)

    storage = RunStorage()
    run_id = storage.save_run(f"{csv_path} -> {target_column}", profile, plan, result, narration)
    print(f"\nSaved as run #{run_id}")


if __name__ == "__main__":
    main()
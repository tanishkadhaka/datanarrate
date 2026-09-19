"""
Full pipeline: real dataset -> profile -> preprocessing decisions ->
4-model comparison with full metrics -> narration -> storage.
"""
import json
from src.data_loader import load_telco_churn
from src.profiler import DataProfiler
from src.preprocessor import PreprocessingDecider
from src.model_selector import ModelSelector
from src.narrator import Narrator
from src.storage import RunStorage


if __name__ == "__main__":
    df = load_telco_churn()

    profile = DataProfiler(df, target_column="Churn").profile()
    print("=== PROFILE ===")
    print(f"{profile.n_rows} rows, {profile.n_cols} cols, imbalanced={profile.is_imbalanced}")

    plan = PreprocessingDecider(profile).decide()
    print("\n=== PREPROCESSING STEP TRACE ===")
    for step in plan.step_trace:
        print("-", step)

    print("\n=== MODEL SELECTION (1-3 minutes on 7k rows, 4 models) ===")
    result = ModelSelector(df, profile, plan).select()
    print(json.dumps(result.to_dict(), indent=2))

    print("\n=== FINAL JUSTIFICATION ===")
    print(result.final_justification)

    print("\n=== NARRATION ===")
    narration = Narrator().narrate(profile, plan, result)
    print(narration)

    storage = RunStorage()
    run_id = storage.save_run("telco_churn", profile, plan, result, narration)
    print(f"\nSaved as run #{run_id}")
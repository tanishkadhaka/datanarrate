from src.data_loader import load_telco_churn
from src.profiler import DataProfiler
from src.preprocessor import PreprocessingDecider
from src.ablation import run_ablation


def test_ablation_runs_all_variants():
    df = load_telco_churn().sample(400, random_state=42)
    profile = DataProfiler(df, target_column="Churn").profile()
    plan = PreprocessingDecider(profile).decide()
    results = run_ablation(df, profile, plan, "logistic_regression", "accuracy", cv_folds=3)
    assert len(results) == 4
    names = [r.variant_name for r in results]
    assert "full_pipeline" in names


def test_full_pipeline_has_zero_drop_from_itself():
    df = load_telco_churn().sample(400, random_state=42)
    profile = DataProfiler(df, target_column="Churn").profile()
    plan = PreprocessingDecider(profile).decide()
    results = run_ablation(df, profile, plan, "logistic_regression", "accuracy", cv_folds=3)
    full = next(r for r in results if r.variant_name == "full_pipeline")
    assert full.score_drop_from_full == 0.0
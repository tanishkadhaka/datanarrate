from src.data_loader import load_telco_churn
from src.profiler import DataProfiler
from src.preprocessor import PreprocessingDecider
from src.ablation import run_ablation
from src.model_selector import ModelSelector
import numpy as np
import pandas as pd


def test_ablation_runs_all_variants():
    df = load_telco_churn().sample(400, random_state=42)
    profile = DataProfiler(df, target_column="Churn").profile()
    plan = PreprocessingDecider(profile).decide()
    results = run_ablation(df, profile, plan, "logistic_regression", "accuracy", cv_folds=3)
    assert len(results) == 5   # four original variants + no_outlier_clipping_R8
    names = [r.variant_name for r in results]
    assert "full_pipeline" in names


def test_full_pipeline_has_zero_drop_from_itself():
    df = load_telco_churn().sample(400, random_state=42)
    profile = DataProfiler(df, target_column="Churn").profile()
    plan = PreprocessingDecider(profile).decide()
    results = run_ablation(df, profile, plan, "logistic_regression", "accuracy", cv_folds=3)
    full = next(r for r in results if r.variant_name == "full_pipeline")
    assert full.score_drop_from_full == 0.0


def make_outlier_df():
    rng = np.random.default_rng(3)
    n = 300
    x = rng.normal(0, 1, n)
    x[:12] = 300   # 4% extreme values
    return pd.DataFrame({"x": x, "y": rng.normal(0, 1, n), "label": (rng.normal(0, 1, n) > 0).astype(int)})


def test_ablation_has_outlier_variant():
    df = make_outlier_df()
    profile = DataProfiler(df, target_column="label").profile()
    plan = PreprocessingDecider(profile).decide()
    results = run_ablation(df, profile, plan, "logistic_regression", "accuracy", cv_folds=3)
    names = [r.variant_name for r in results]
    assert "no_outlier_clipping_R8" in names


def test_ablation_full_pipeline_matches_model_selector():
    # ablation now uses the selector's model settings, so the full pipeline scores the same
    df = make_outlier_df()
    profile = DataProfiler(df, target_column="label").profile()
    plan = PreprocessingDecider(profile).decide()
    result = ModelSelector(df, profile, plan).select(cv_folds=3)
    results = run_ablation(df, profile, plan, result.best_model_name, result.metric_used,
                           cv_folds=result.cv_folds_used)
    full = next(r for r in results if r.variant_name == "full_pipeline")
    assert abs(full.score - result.best_score) < 1e-9


def test_ablation_can_run_only_the_outlier_variant():
    df = make_outlier_df()
    profile = DataProfiler(df, target_column="label").profile()
    plan = PreprocessingDecider(profile).decide()
    results = run_ablation(df, profile, plan, "logistic_regression", "accuracy", cv_folds=3,
                           variant_names=["no_outlier_clipping_R8"])
    assert [r.variant_name for r in results] == ["full_pipeline", "no_outlier_clipping_R8"]

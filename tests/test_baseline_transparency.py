import pytest

from src.data_loader import load_telco_churn
from src.profiler import DataProfiler
from src.preprocessor import PreprocessingDecider
from src.model_selector import ModelSelector


def get_result():
    df = load_telco_churn().sample(500, random_state=42)
    profile = DataProfiler(df, target_column="Churn").profile()
    plan = PreprocessingDecider(profile).decide()
    return ModelSelector(df, profile, plan).select(cv_folds=3)


def test_baseline_info_is_populated():
    info = get_result().baseline_info
    assert info is not None
    assert "Logistic regression" in info.summary
    assert len(info.steps) >= 6
    assert len(info.why_chosen) >= 3


def test_each_stage_compares_both_sides():
    for step in get_result().baseline_info.steps:
        assert "Stage" in step
        assert "Baseline (fixed recipe)" in step
        assert "DataNarrate (decided from the data)" in step


def test_ladder_starts_at_baseline_and_ends_at_best_score():
    result = get_result()
    assert len(result.ladder) == 4
    assert result.ladder[0].score == pytest.approx(result.baseline_score)
    assert result.ladder[-1].score == pytest.approx(result.best_score)
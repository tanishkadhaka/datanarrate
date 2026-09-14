import pandas as pd
from sklearn.datasets import load_breast_cancer
from src.profiler import DataProfiler
from src.preprocessor import PreprocessingDecider
from src.model_selector import ModelSelector, CANDIDATE_MODELS


def get_result():
    data = load_breast_cancer(as_frame=True)
    df = data.frame
    profile = DataProfiler(df, target_column="target").profile()
    plan = PreprocessingDecider(profile).decide()
    # cv_folds=3 here keeps the test suite fast; production runs use 5
    return ModelSelector(df, profile, plan).select(cv_folds=3)


def test_ten_candidate_models_registered():
    assert len(CANDIDATE_MODELS) == 10


def test_selects_a_best_model():
    result = get_result()
    assert result.best_model_name in CANDIDATE_MODELS.keys()


def test_all_candidates_scored():
    result = get_result()
    assert len(result.candidate_results) == 10


def test_scores_are_valid_range():
    result = get_result()
    for r in result.candidate_results:
        assert 0.0 <= r.mean_score <= 1.0


def test_metric_selection_binary_balanced_uses_accuracy():
    result = get_result()
    assert result.metric_used == "accuracy"


def test_baseline_score_is_valid():
    result = get_result()
    assert 0.0 <= result.baseline_score <= 1.0


def test_explanations_generated_for_every_model():
    result = get_result()
    # one explanation for the winner + one for each of the other 9 candidates
    assert len(result.explanations) == 10


def test_winner_mentioned_in_first_explanation():
    result = get_result()
    assert result.best_model_name in result.explanations[0]
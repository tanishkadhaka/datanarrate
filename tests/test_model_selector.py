from src.data_loader import load_telco_churn
from src.profiler import DataProfiler
from src.preprocessor import PreprocessingDecider
from src.model_selector import ModelSelector, _build_candidates


def get_result():
    df = load_telco_churn().sample(500, random_state=42)  # small sample keeps tests fast
    profile = DataProfiler(df, target_column="Churn").profile()
    plan = PreprocessingDecider(profile).decide()
    return ModelSelector(df, profile, plan).select(cv_folds=3)


def test_four_candidate_models():
    assert len(_build_candidates()) == 4


def test_all_candidates_scored():
    result = get_result()
    assert len(result.candidate_results) == 4


def test_every_model_has_confusion_matrix():
    result = get_result()
    for r in result.candidate_results:
        assert len(r.confusion_matrix) == 2  # binary target


def test_every_model_has_full_metrics():
    result = get_result()
    for r in result.candidate_results:
        assert 0.0 <= r.accuracy <= 1.0
        assert 0.0 <= r.precision <= 1.0
        assert 0.0 <= r.recall <= 1.0
        assert 0.0 <= r.f1 <= 1.0


def test_voting_ensemble_is_a_candidate():
    result = get_result()
    names = [r.model_name for r in result.candidate_results]
    assert "voting_ensemble" in names


def test_final_justification_mentions_winner():
    result = get_result()
    assert result.best_model_name in result.final_justification


def test_baseline_score_is_valid():
    result = get_result()
    assert 0.0 <= result.baseline_score <= 1.0
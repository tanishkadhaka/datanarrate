from unittest.mock import MagicMock, patch
import pandas as pd
from sklearn.datasets import load_breast_cancer

from src.profiler import DataProfiler
from src.preprocessor import PreprocessingDecider
from src.model_selector import ModelSelector
from src.narrator import Narrator


def get_pipeline_outputs():
    data = load_breast_cancer(as_frame=True)
    df = data.frame
    profile = DataProfiler(df, target_column="target").profile()
    plan = PreprocessingDecider(profile).decide()
    result = ModelSelector(df, profile, plan).select(cv_folds=3)
    return profile, plan, result


def test_build_prompt_includes_best_model():
    profile, plan, result = get_pipeline_outputs()
    narrator = Narrator(api_key="fake_key_for_prompt_test")
    prompt = narrator.build_prompt(profile, plan, result)
    assert result.best_model_name in prompt


def test_build_prompt_includes_baseline_score():
    profile, plan, result = get_pipeline_outputs()
    narrator = Narrator(api_key="fake_key_for_prompt_test")
    prompt = narrator.build_prompt(profile, plan, result)
    assert str(round(result.baseline_score, 4)) in prompt


def test_narrate_calls_api_and_returns_text():
    profile, plan, result = get_pipeline_outputs()
    narrator = Narrator(api_key="fake_key_for_prompt_test")

    mock_response = MagicMock()
    mock_response.choices[0].message.content = "This is a test narration."

    with patch.object(narrator.client.chat.completions, "create", return_value=mock_response) as mock_create:
        narration = narrator.narrate(profile, plan, result)
        assert narration == "This is a test narration."
        mock_create.assert_called_once()


def test_missing_api_key_raises():
    import pytest
    with patch.dict("os.environ", {}, clear=True):
        with pytest.raises(ValueError):
            Narrator(api_key=None)
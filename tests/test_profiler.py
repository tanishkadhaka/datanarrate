import pandas as pd
import pytest
from src.profiler import DataProfiler


def make_df():
    return pd.DataFrame({
        "age": [25, 30, 22, None, 40, 35],
        "city": ["A", "B", "A", "C", "B", "A"],
        "label": [0, 1, 0, 1, 0, 1],
    })


def test_basic_shape():
    profile = DataProfiler(make_df(), target_column="label").profile()
    assert profile.n_rows == 6
    assert profile.n_cols == 3


def test_missing_value_detection():
    profile = DataProfiler(make_df(), target_column="label").profile()
    age_col = next(c for c in profile.columns if c.name == "age")
    assert age_col.missing_pct > 0


def test_target_excluded_from_columns():
    profile = DataProfiler(make_df(), target_column="label").profile()
    assert "label" not in [c.name for c in profile.columns]


def test_invalid_target_raises():
    with pytest.raises(ValueError):
        DataProfiler(make_df(), target_column="doesnt_exist")


def test_balanced_classes_not_flagged():
    profile = DataProfiler(make_df(), target_column="label").profile()
    assert profile.is_imbalanced is False
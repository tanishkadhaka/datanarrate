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

# --- outlier stats (IQR rule) ---

def test_outlier_stats_on_hand_made_column():
    # 1..11 plus 100: Q1 = 3.75, Q3 = 9.25, IQR = 5.5, fences = [-4.5, 17.5]
    df = pd.DataFrame({"x": list(range(1, 12)) + [100], "label": [0, 1] * 6})
    col = DataProfiler(df, target_column="label").profile().columns[0]
    assert col.outlier_count == 1
    assert col.outlier_pct == 8.33
    assert col.lower_fence == pytest.approx(-4.5)
    assert col.upper_fence == pytest.approx(17.5)
    assert col.iqr_is_zero is False


def test_iqr_is_zero_flag():
    df = pd.DataFrame({"x": [0] * 9 + [5], "label": [0, 1] * 5})
    col = DataProfiler(df, target_column="label").profile().columns[0]
    assert col.iqr_is_zero is True
    assert col.outlier_pct == 10.0   # the IQR fences flag the one non-zero value


def test_outlier_stats_ignore_missing_values():
    df = pd.DataFrame({"x": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, None, 1000], "label": [0, 1] * 6})
    col = DataProfiler(df, target_column="label").profile().columns[0]
    assert col.outlier_count == 1
    assert col.outlier_pct == round(1 / 11 * 100, 2)   # 11 values present, not 12


def test_categorical_and_target_have_no_outlier_stats():
    profile = DataProfiler(make_df(), target_column="label").profile()
    city = next(c for c in profile.columns if c.name == "city")
    assert city.lower_fence is None and city.outlier_count == 0
    assert "label" not in [c.name for c in profile.columns]

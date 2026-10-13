import numpy as np
import pandas as pd
import pytest

from src.target_preparer import prepare_target, InvalidTargetError


def test_clean_binary_target_is_left_alone():
    df = pd.DataFrame({"x": range(100), "y": ["a", "b"] * 50})
    out, notes = prepare_target(df, "y")
    assert set(out["y"].unique()) == {"a", "b"}
    assert len(out) == 100


def test_continuous_numeric_target_is_binned():
    df = pd.DataFrame({"x": np.random.RandomState(0).rand(200), "age": np.arange(200)})
    out, notes = prepare_target(df, "age")
    assert out["age"].nunique() == 4
    assert any("continuous" in n for n in notes)


def test_mostly_zero_numeric_target_split_into_zero_vs_nonzero():
    values = [0] * 180 + list(range(1, 21))
    df = pd.DataFrame({"x": np.arange(200) % 9, "gain": values})
    out, notes = prepare_target(df, "gain")
    assert set(out["gain"].unique()) == {"equals 0", "not 0"}


def test_rare_classes_are_merged():
    labels = ["a"] * 60 + ["b"] * 30 + ["c"] * 5 + ["d"] * 5
    df = pd.DataFrame({"x": np.arange(100) % 7, "y": labels})
    out, notes = prepare_target(df, "y")
    assert "Other (merged)" in set(out["y"].unique())
    assert out["y"].nunique() == 3


def test_identifier_like_target_rejected():
    df = pd.DataFrame({"id": [f"id_{i}" for i in range(50)], "x": range(50)})
    with pytest.raises(InvalidTargetError):
        prepare_target(df, "id")


def test_feature_that_duplicates_target_is_dropped():
    y = ["low", "mid", "high"] * 40
    code = [1, 2, 3] * 40
    df = pd.DataFrame({"y": y, "y_code": code, "noise": np.random.RandomState(1).rand(120)})
    out, notes = prepare_target(df, "y")
    assert "y_code" not in out.columns
    assert any("y_code" in n for n in notes)


def test_class_order_is_reported():
    df = pd.DataFrame({"x": range(40), "y": ["no", "yes"] * 20})
    out, notes = prepare_target(df, "y")
    assert any("0 = no" in n for n in notes)


def test_rows_with_missing_target_are_removed():
    df = pd.DataFrame({"x": range(40), "y": ["a", "b"] * 19 + [None, None]})
    out, notes = prepare_target(df, "y")
    assert len(out) == 38
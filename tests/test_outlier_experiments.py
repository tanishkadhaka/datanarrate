import numpy as np
import pandas as pd
import pytest

from run_outlier_injection import inject_outliers
from run_outlier_comparison import compare_dataset, verdict, choose_metric
from src.profiler import DataProfiler
from src.model_selector import ModelSelector
from src.preprocessor import PreprocessingDecider


def make_df(n=300):
    rng = np.random.default_rng(5)
    x = rng.normal(10, 2, n)
    y = rng.normal(0, 1, n)
    z = rng.normal(0, 1, n)
    label = (y + rng.normal(0, 0.5, n) > 0).astype(int)
    return pd.DataFrame({"x": x, "y": y, "z": z, "label": label})


def test_injection_puts_extremes_in_the_chosen_columns_only():
    df = make_df(200)
    new_df, positions = inject_outliers(df, ["x", "y"], rate=0.05, seed=0)

    for col in ["x", "y"]:
        assert len(positions[col]) == 10   # 5% of 200 rows
        mean, std = df[col].mean(), df[col].std()
        injected = new_df[col].to_numpy()[positions[col]]
        assert np.allclose(np.abs(injected - mean), 10 * std)   # mean + or - 10 std
        # every other row is untouched
        others = np.setdiff1d(np.arange(200), positions[col])
        assert np.array_equal(new_df[col].to_numpy()[others], df[col].to_numpy()[others])

    assert new_df["z"].equals(df["z"])               # column not chosen
    assert new_df["label"].equals(df["label"])       # target not touched
    assert df["x"].max() < 20                        # the original dataframe is unchanged


def test_injection_with_zero_rate_changes_nothing():
    df = make_df(100)
    new_df, positions = inject_outliers(df, ["x"], rate=0.0)
    assert new_df.equals(df)
    assert len(positions["x"]) == 0


def test_comparison_runs_on_a_tiny_dataset():
    df = make_df(300)
    df.loc[:11, "x"] = 500   # 4% extreme values so that R8 clips something
    rows = compare_dataset("tiny", df, "label", seeds=[0, 1])

    assert [r["model"] for r in rows] == ["logistic_regression", "random_forest",
                                          "gradient_boosting", "voting_ensemble"]
    for r in rows:
        assert r["n_seeds"] == 2
        assert r["columns_clipped"] >= 1
        assert 0.0 <= r["mean_with"] <= 1.0 and 0.0 <= r["mean_without"] <= 1.0
        assert r["verdict"] in ("better", "worse", "within 1 std", "identical")

    # the metric rule used here is the same one the model selector uses
    profile = DataProfiler(df, "label").profile()
    plan = PreprocessingDecider(profile).decide()
    assert choose_metric(profile) == ModelSelector(df, profile, plan).select(cv_folds=3).metric_used


def test_verdict_compares_difference_to_one_std():
    without = [0.80, 0.82, 0.78, 0.81, 0.79]   # std is about 0.016
    assert verdict([x + 0.005 for x in without], without) == "within 1 std"
    assert verdict([x + 0.05 for x in without], without) == "better"
    assert verdict([x - 0.05 for x in without], without) == "worse"
    assert verdict(without, without) == "identical"

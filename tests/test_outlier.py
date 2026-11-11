import numpy as np
import pandas as pd
import pytest
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.outlier import OutlierClipper, build_clipper
from src.profiler import DataProfiler
from src.preprocessor import PreprocessingDecider
from src.model_selector import ModelSelector, build_decided_preprocessor


def test_clipper_learns_bounds_from_fit_data_only():
    train = np.arange(1, 101, dtype=float).reshape(-1, 1)   # Q1=25.75, Q3=75.25, IQR=49.5
    clipper = OutlierClipper(methods=["iqr"]).fit(train)
    assert clipper.upper_[0] == pytest.approx(149.5)
    assert clipper.lower_[0] == pytest.approx(-48.5)

    # test values beyond the TRAIN bounds are clipped to the train bounds
    test = np.array([[5000.0], [-5000.0], [50.0]])
    out = clipper.transform(test)
    assert out.ravel().tolist() == pytest.approx([149.5, -48.5, 50.0])

    # transforming new data must not change the learned bounds
    assert clipper.upper_[0] == pytest.approx(149.5)


def test_clipper_ignores_test_rows_when_fitting():
    # the extreme row is in the "test" part, so it must not influence the bounds
    values = np.append(np.arange(1, 101, dtype=float), 1_000_000.0).reshape(-1, 1)
    train, test = values[:100], values[100:]
    clipper = OutlierClipper(methods=["iqr"]).fit(train)
    assert clipper.upper_[0] == pytest.approx(149.5)
    assert clipper.transform(test)[0, 0] == pytest.approx(149.5)

    # fitting on everything gives different bounds, which is the leak we avoid
    leaky = OutlierClipper(methods=["iqr"]).fit(values)
    assert leaky.upper_[0] != clipper.upper_[0]


def test_clipper_methods_per_column():
    X = np.column_stack([np.arange(1, 101, dtype=float)] * 3)
    clipper = OutlierClipper(methods=["iqr", "percentile", "none"]).fit(X)
    assert clipper.upper_[0] == pytest.approx(149.5)
    assert clipper.upper_[1] == pytest.approx(np.percentile(X[:, 1], 99))
    assert np.isinf(clipper.upper_[2])


def test_clipper_keeps_missing_values():
    X = np.array([[1.0], [2.0], [np.nan], [3.0], [4.0], [1000.0]])
    out = OutlierClipper(methods=["iqr"]).fit_transform(X)
    assert np.isnan(out[2, 0])
    assert out.shape == X.shape   # no rows dropped


def test_build_clipper_returns_none_when_nothing_to_clip():
    df = pd.DataFrame({"x": list(range(1, 11)), "label": [0, 1] * 5})
    plan = PreprocessingDecider(DataProfiler(df, "label").profile()).decide()
    decisions = {d.column: d for d in plan.column_decisions}
    assert build_clipper(decisions, ["x"]) is None


def make_outlier_dataset():
    rng = np.random.default_rng(0)
    n = 300
    income = rng.normal(50, 10, n)
    income[:12] = 1000   # 4% extreme values
    other = rng.normal(0, 1, n)
    label = (other + rng.normal(0, 0.5, n) > 0).astype(int)
    return pd.DataFrame({"income": income, "other": other, "label": label})


def test_pipeline_runs_end_to_end_with_outlier_column():
    df = make_outlier_dataset()
    profile = DataProfiler(df, "label").profile()
    plan = PreprocessingDecider(profile).decide()
    income = next(d for d in plan.column_decisions if d.column == "income")
    assert income.outlier_action == "clip"

    selector = ModelSelector(df, profile, plan)
    result = selector.select(cv_folds=3)
    assert len(result.candidate_results) == 4
    assert 0.0 <= result.best_score <= 1.0

    # the clipper is part of the numeric branch and runs before the scaler
    pre = build_decided_preprocessor(plan, df.drop(columns=["label"]))
    numeric_steps = [name for name, _ in pre.transformers[0][1].steps]
    assert numeric_steps == ["clip", "impute", "scale"]


def test_clipper_inside_pipeline_learns_bounds_per_fit():
    # same check through a Pipeline, which is how cross-validation uses it
    values = np.append(np.arange(1, 101, dtype=float), 1_000_000.0).reshape(-1, 1)
    pipe = Pipeline([("clip", OutlierClipper(methods=["iqr"])), ("scale", StandardScaler())])
    pipe.fit(values[:100])
    assert pipe.named_steps["clip"].upper_[0] == pytest.approx(149.5)
    scaled_test = pipe.transform(values[100:])
    # a clipped test value sits at the scaled train maximum, not at an enormous number
    assert scaled_test[0, 0] < 10

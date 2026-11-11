import pytest
import pandas as pd
from src.profiler import DataProfiler
from src.preprocessor import PreprocessingDecider


def make_profile(df, target="label"):
    return DataProfiler(df, target_column=target).profile()


def test_high_missing_column_dropped():
    df = pd.DataFrame({
        "mostly_missing": [None] * 8 + [1, 2],
        "clean": list(range(10)),
        "label": [0, 1] * 5,
    })
    plan = PreprocessingDecider(make_profile(df)).decide()
    d = next(x for x in plan.column_decisions if x.column == "mostly_missing")
    assert d.missing_action == "drop_column"


def test_low_missing_numeric_imputed_with_median():
    df = pd.DataFrame({
        "age": [25, 30, None, 40, 35, 28, 33, 31, 29, 27],
        "label": [0, 1] * 5,
    })
    plan = PreprocessingDecider(make_profile(df)).decide()
    d = next(x for x in plan.column_decisions if x.column == "age")
    assert d.missing_action == "impute_median"


def test_low_cardinality_categorical_one_hot():
    df = pd.DataFrame({
        "city": ["A", "B", "A", "C", "B", "A", "C", "B", "A", "C"],
        "label": [0, 1] * 5,
    })
    plan = PreprocessingDecider(make_profile(df)).decide()
    d = next(x for x in plan.column_decisions if x.column == "city")
    assert d.encoding_action == "one_hot"


def test_high_cardinality_categorical_frequency_encoded():
    df = pd.DataFrame({
        "region_code": [f"r{i}" for i in range(25)] + [f"r{i}" for i in range(5)],
        "label": [0, 1] * 15,
    })
    plan = PreprocessingDecider(make_profile(df)).decide()
    d = next(x for x in plan.column_decisions if x.column == "region_code")
    assert d.encoding_action == "frequency_encode"


def test_identifier_column_dropped():
    df = pd.DataFrame({
        "customer_id": [f"cust_{i}" for i in range(20)],
        "score": list(range(20)),
        "label": [0, 1] * 10,
    })
    plan = PreprocessingDecider(make_profile(df)).decide()
    d = next(x for x in plan.column_decisions if x.column == "customer_id")
    assert d.missing_action == "drop_column"


def test_numeric_column_scheduled_for_scaling():
    df = pd.DataFrame({"score": list(range(10)), "label": [0, 1] * 5})
    plan = PreprocessingDecider(make_profile(df)).decide()
    d = next(x for x in plan.column_decisions if x.column == "score")
    assert d.scaling_action == "standard_scale"


def test_step_trace_has_an_entry_per_reason():
    df = pd.DataFrame({"score": list(range(10)), "label": [0, 1] * 5})
    plan = PreprocessingDecider(make_profile(df)).decide()
    assert len(plan.step_trace) >= 1
    assert plan.step_trace[0].startswith("[score]")

# --- rule IDs and the outlier rule (R8) ---

from src.profiler import ColumnProfile, DatasetProfile


def decide_one_numeric(values):
    df = pd.DataFrame({"x": values, "label": [0, 1] * (len(values) // 2)})
    plan = PreprocessingDecider(make_profile(df)).decide()
    return next(d for d in plan.column_decisions if d.column == "x")


def test_few_outliers_means_no_clipping():
    # one extreme value in 200 = 0.5%, below the 1% threshold
    d = decide_one_numeric(list(range(1, 200)) + [100000])
    assert d.outlier_action == "none"
    assert "too few outliers" in d.reasons[-2]
    assert d.reasons[-2].startswith("R8")


def test_many_outliers_clip_to_iqr_fences():
    # 5 extreme values in 100 = 5%
    d = decide_one_numeric(list(range(1, 96)) + [1000] * 5)
    assert d.outlier_action == "clip"
    assert d.outlier_bounds == "iqr"
    assert d.clip_upper == pytest.approx(149.5)
    assert "5.0% of values fall outside" in d.reasons[-2]


def test_zero_iqr_falls_back_to_percentiles():
    # mostly zeros (like capital-gain): the IQR is 0 so IQR fences would flag every non-zero
    d = decide_one_numeric([0] * 180 + list(range(1, 21)))
    assert d.outlier_action == "clip"
    assert d.outlier_bounds == "percentile"
    assert d.clip_lower == 0
    assert "1st and 99th percentiles" in d.reasons[-2]


def test_zero_iqr_with_equal_percentiles_is_left_alone():
    col = ColumnProfile(name="x", dtype="numeric", missing_pct=0.0, n_unique=5,
                        is_high_cardinality=False, outlier_count=20, outlier_pct=2.0,
                        lower_fence=0.0, upper_fence=0.0, iqr_is_zero=True, p01=3.0, p99=3.0)
    profile = DatasetProfile(n_rows=1000, n_cols=2, target_column="y", task_type="classification",
                             n_classes=2, class_distribution={}, is_imbalanced=False, columns=[col])
    d = PreprocessingDecider(profile).decide().column_decisions[0]
    assert d.outlier_action == "none"
    assert "no sensible range" in d.reasons[-2]


def test_binary_numeric_column_is_not_clipped():
    d = decide_one_numeric([0] * 90 + [1] * 10)
    assert d.outlier_action == "none"


def test_dropped_columns_get_no_outlier_rule():
    df = pd.DataFrame({
        "mostly_missing": [None] * 8 + [1, 1000],
        "customer_id": [f"c{i}" for i in range(10)],
        "label": [0, 1] * 5,
    })
    plan = PreprocessingDecider(make_profile(df)).decide()
    for d in plan.column_decisions:
        assert d.outlier_action == "none"
        assert not any(r.startswith("R8") for r in d.reasons)


def test_every_reason_starts_with_a_rule_id():
    df = pd.DataFrame({
        "age": [25, 30, None, 40, 35, 28, 33, 31, 29, 27],
        "city": ["A", "B", "A", "C", "B", "A", "C", "B", "A", "C"],
        "label": [0, 1] * 5,
    })
    plan = PreprocessingDecider(make_profile(df)).decide()
    for d in plan.column_decisions:
        for reason in d.reasons:
            assert reason[:2] in ("R1", "R2", "R3", "R4", "R5", "R6", "R7", "R8")

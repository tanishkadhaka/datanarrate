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
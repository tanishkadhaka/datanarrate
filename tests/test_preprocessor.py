import pandas as pd
from src.profiler import DataProfiler
from src.preprocessor import PreprocessingDecider


def make_profile(df, target="label"):
    return DataProfiler(df, target_column=target).profile()


def test_high_missing_column_dropped():
    df = pd.DataFrame({
        "mostly_missing": [None] * 8 + [1, 2],
        "clean": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        "label": [0, 1] * 5,
    })
    plan = PreprocessingDecider(make_profile(df)).decide()
    decision = next(d for d in plan.column_decisions if d.column == "mostly_missing")
    assert decision.missing_action == "drop_column"


def test_low_missing_numeric_imputed_with_median():
    df = pd.DataFrame({
        "age": [25, 30, None, 40, 35, 28, 33, 31, 29, 27],
        "label": [0, 1] * 5,
    })
    plan = PreprocessingDecider(make_profile(df)).decide()
    decision = next(d for d in plan.column_decisions if d.column == "age")
    assert decision.missing_action == "impute_median"


def test_low_cardinality_categorical_one_hot():
    df = pd.DataFrame({
        "city": ["A", "B", "A", "C", "B", "A", "C", "B", "A", "C"],
        "label": [0, 1] * 5,
    })
    plan = PreprocessingDecider(make_profile(df)).decide()
    decision = next(d for d in plan.column_decisions if d.column == "city")
    assert decision.encoding_action == "one_hot"


def test_high_cardinality_categorical_frequency_encoded():
    df = pd.DataFrame({
        "user_id": [f"user_{i}" for i in range(30)],
        "label": [0, 1] * 15,
    })
    plan = PreprocessingDecider(make_profile(df)).decide()
    decision = next(d for d in plan.column_decisions if d.column == "user_id")
    assert decision.encoding_action == "frequency_encode"


def test_numeric_column_scheduled_for_scaling():
    df = pd.DataFrame({
        "score": list(range(10)),
        "label": [0, 1] * 5,
    })
    plan = PreprocessingDecider(make_profile(df)).decide()
    decision = next(d for d in plan.column_decisions if d.column == "score")
    assert decision.scaling_action == "standard_scale"


def test_every_active_decision_has_a_reason():
    df = pd.DataFrame({
        "score": list(range(10)),
        "label": [0, 1] * 5,
    })
    plan = PreprocessingDecider(make_profile(df)).decide()
    for decision in plan.column_decisions:
        has_action = (
            decision.missing_action != "none"
            or decision.encoding_action != "none"
            or decision.scaling_action != "none"
        )
        if has_action:
            assert len(decision.reasons) > 0
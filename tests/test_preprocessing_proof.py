import numpy as np
import pandas as pd

from src.profiler import DataProfiler
from src.preprocessor import PreprocessingDecider
from src.preprocessing_proof import build_preprocessing_proof


def make_proof():
    rng = np.random.RandomState(0)
    n = 200
    df = pd.DataFrame({
        "cust_id": [f"c{i}" for i in range(n)],
        "age": np.where(rng.rand(n) < 0.05, np.nan, rng.randint(18, 80, n)).astype(float),
        "city": rng.choice(["A", "B", "C"], n),
        "zone": rng.choice([f"z{i}" for i in range(25)], n),
        "label": rng.choice([0, 1], n),
    })
    df.loc[:4, "city"] = "?"
    profile = DataProfiler(df, target_column="label").profile()
    plan = PreprocessingDecider(profile).decide()
    return build_preprocessing_proof(df, profile, plan), plan


def _col(proof, name):
    return next(c for c in proof.columns if c.column == name)


def test_every_column_has_a_proof():
    proof, plan = make_proof()
    assert len(proof.columns) == len(plan.column_decisions)


def test_all_checks_pass():
    proof, _ = make_proof()
    assert proof.full_check_passed


def test_no_missing_cells_after_preprocessing():
    proof, _ = make_proof()
    assert proof.nan_cells_after == 0


def test_numeric_proof_shows_mean_zero_std_one():
    proof, _ = make_proof()
    age = _col(proof, "age")
    assert abs(age.after["mean"]) < 1e-3
    assert abs(age.after["std"] - 1) < 1e-3


def test_one_hot_output_matches_category_count():
    proof, _ = make_proof()
    assert _col(proof, "city").after["output columns"] == 4


def test_dropped_identifier_has_evidence():
    proof, _ = make_proof()
    cust = _col(proof, "cust_id")
    assert any("Identifier rule" in name for name, _, _ in cust.checks)


def test_placeholder_values_are_flagged():
    proof, _ = make_proof()
    city = _col(proof, "city")
    assert any("'?'" in w for w in city.warnings)


def test_frequency_encoding_checks_pass():
    proof, _ = make_proof()
    assert _col(proof, "zone").all_passed
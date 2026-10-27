import json
import sqlite3

import numpy as np
import pandas as pd

from src.profiler import DataProfiler
from src.preprocessor import PreprocessingDecider
from src.model_selector import ModelSelector
from src.storage import RunStorage


def make_run():
    rng = np.random.default_rng(1)
    n = 120
    x = rng.normal(0, 1, n)
    x[:6] = 500   # 5% outliers
    df = pd.DataFrame({"x": x, "y": rng.normal(0, 1, n), "label": (rng.normal(0, 1, n) > 0).astype(int)})
    profile = DataProfiler(df, "label").profile()
    plan = PreprocessingDecider(profile).decide()
    result = ModelSelector(df, profile, plan).select(cv_folds=3)
    return profile, plan, result


def test_outlier_decisions_are_saved_with_the_run(tmp_path):
    profile, plan, result = make_run()
    storage = RunStorage(str(tmp_path / "runs.db"))
    run_id = storage.save_run("tiny", profile, plan, result, "narration text")

    stored_plan = storage.get_plan(run_id)
    x = next(d for d in stored_plan.column_decisions if d.column == "x")
    assert x.outlier_action == "clip"
    assert x.clip_upper is not None
    assert any(r.startswith("R8") for r in x.reasons)


def test_old_runs_without_outlier_fields_still_load(tmp_path):
    profile, plan, result = make_run()
    db_path = str(tmp_path / "runs.db")
    storage = RunStorage(db_path)
    run_id = storage.save_run("tiny", profile, plan, result, "narration text")

    # rewrite the stored plan the way it looked before outlier handling existed
    old_plan = plan.to_dict()
    for c in old_plan["column_decisions"]:
        for key in ("outlier_action", "outlier_bounds", "clip_lower", "clip_upper"):
            del c[key]
    with sqlite3.connect(db_path) as conn:
        conn.execute("UPDATE runs SET plan_json = ? WHERE id = ?", (json.dumps(old_plan), run_id))

    stored_plan = storage.get_plan(run_id)
    assert all(d.outlier_action == "none" for d in stored_plan.column_decisions)
    assert len(storage.list_runs()) == 1   # the history table still works

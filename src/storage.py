"""
Module 5 — Storage
Saves each run to SQLite, including full metrics and the justification, so
runs can be compared later (needed for your multi-dataset evaluation).
"""
import sqlite3
import json
from dataclasses import dataclass
from datetime import datetime
from typing import Optional, List

from src.profiler import DatasetProfile
from src.preprocessor import PreprocessingPlan
from src.model_selector import SelectionResult

DB_PATH = "datanarrate.db"


@dataclass
class StoredRun:
    id: int
    timestamp: str
    dataset_name: str
    n_rows: int
    n_cols: int
    best_model_name: str
    best_score: float
    baseline_score: float
    improvement_over_baseline: float
    metric_used: str
    final_justification: str
    narration: str


class RunStorage:
    def __init__(self, db_path: str = DB_PATH):
        self.db_path = db_path
        self._init_schema()

    def _connect(self):
        return sqlite3.connect(self.db_path)

    def _init_schema(self):
        with self._connect() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS runs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    dataset_name TEXT NOT NULL,
                    n_rows INTEGER, n_cols INTEGER,
                    best_model_name TEXT, best_score REAL,
                    baseline_score REAL, improvement_over_baseline REAL,
                    metric_used TEXT, final_justification TEXT, narration TEXT,
                    profile_json TEXT, plan_json TEXT, result_json TEXT
                )
            """)

    def save_run(self, dataset_name, profile: DatasetProfile, plan: PreprocessingPlan,
                 result: SelectionResult, narration: str) -> int:
        with self._connect() as conn:
            cursor = conn.execute(
                """INSERT INTO runs (timestamp, dataset_name, n_rows, n_cols,
                   best_model_name, best_score, baseline_score, improvement_over_baseline,
                   metric_used, final_justification, narration, profile_json, plan_json, result_json)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (datetime.now().isoformat(), dataset_name, profile.n_rows, profile.n_cols,
                 result.best_model_name, result.best_score, result.baseline_score,
                 result.improvement_over_baseline, result.metric_used, result.final_justification,
                 narration, json.dumps(profile.to_dict(), default=str),
                 json.dumps(plan.to_dict(), default=str), json.dumps(result.to_dict(), default=str)),
            )
            return cursor.lastrowid

    def list_runs(self) -> List[StoredRun]:
        with self._connect() as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                """SELECT id, timestamp, dataset_name, n_rows, n_cols, best_model_name,
                   best_score, baseline_score, improvement_over_baseline, metric_used,
                   final_justification, narration FROM runs ORDER BY timestamp DESC"""
            ).fetchall()
            return [StoredRun(**dict(r)) for r in rows]

    def get_run(self, run_id: int) -> Optional[dict]:
        with self._connect() as conn:
            conn.row_factory = sqlite3.Row
            row = conn.execute("SELECT * FROM runs WHERE id = ?", (run_id,)).fetchone()
            return dict(row) if row else None
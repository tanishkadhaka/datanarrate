"""
Module 3 — Model Selection
Builds a preprocessing pipeline from Module 2's decisions, shortlists candidate
classifiers, cross-validates each using the metric appropriate for the dataset,
and compares the winner against a fixed no-decision baseline pipeline.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score

from src.profiler import DatasetProfile
from src.preprocessor import PreprocessingPlan


class FrequencyEncoder(BaseEstimator, TransformerMixin):
    """Encodes each category as its frequency (proportion) in the training data.
    Used instead of one-hot for high-cardinality columns to avoid dimensionality blowup."""

    def fit(self, X, y=None):
        X = pd.DataFrame(X)
        self.freq_maps_ = [col.value_counts(normalize=True).to_dict() for _, col in X.items()]
        return self

    def transform(self, X):
        X = pd.DataFrame(X)
        out = np.zeros(X.shape)
        for i, (_, col) in enumerate(X.items()):
            out[:, i] = col.map(self.freq_maps_[i]).fillna(0.0).values
        return out


@dataclass
class ModelResult:
    model_name: str
    mean_score: float
    std_score: float


@dataclass
class SelectionResult:
    metric_used: str
    dropped_columns: List[str]
    candidate_results: List[ModelResult]
    best_model_name: str
    best_score: float
    baseline_score: float
    improvement_over_baseline: float

    def to_dict(self) -> dict:
        return {
            "metric_used": self.metric_used,
            "dropped_columns": self.dropped_columns,
            "candidate_results": [vars(r) for r in self.candidate_results],
            "best_model_name": self.best_model_name,
            "best_score": round(self.best_score, 4),
            "baseline_score": round(self.baseline_score, 4),
            "improvement_over_baseline": round(self.improvement_over_baseline, 4),
        }


CANDIDATE_MODELS = {
    "logistic_regression": LogisticRegression(max_iter=1000),
    "random_forest": RandomForestClassifier(n_estimators=200, random_state=42),
    "gradient_boosting": GradientBoostingClassifier(random_state=42),
}


class ModelSelector:
    """Builds pipelines from the profile + plan, cross-validates candidates,
    and compares the winner against a fixed baseline."""

    def __init__(self, df: pd.DataFrame, profile: DatasetProfile, plan: PreprocessingPlan):
        self.df = df
        self.profile = profile
        self.plan = plan
        self.target = profile.target_column

    def select(self, cv_folds: int = 5, random_state: int = 42) -> SelectionResult:
        X_full = self.df.drop(columns=[self.target])
        y = self.df[self.target]

        dropped = [d.column for d in self.plan.column_decisions if d.missing_action == "drop_column"]
        X = X_full.drop(columns=dropped)

        metric = "f1_macro" if (self.profile.is_imbalanced or self.profile.n_classes > 2) else "accuracy"

        pipeline_preprocessor = self._build_decided_preprocessor(X)
        cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)

        candidate_results = []
        best_name, best_score, best_pipeline = None, -np.inf, None

        for name, model in CANDIDATE_MODELS.items():
            pipe = Pipeline([("preprocess", pipeline_preprocessor), ("model", model)])
            scores = cross_val_score(pipe, X, y, cv=cv, scoring=metric)
            candidate_results.append(ModelResult(
                model_name=name,
                mean_score=float(scores.mean()),
                std_score=float(scores.std()),
            ))
            if scores.mean() > best_score:
                best_name, best_score, best_pipeline = name, scores.mean(), pipe

        baseline_score = self._run_baseline(X_full, y, metric, cv)

        return SelectionResult(
            metric_used=metric,
            dropped_columns=dropped,
            candidate_results=candidate_results,
            best_model_name=best_name,
            best_score=float(best_score),
            baseline_score=float(baseline_score),
            improvement_over_baseline=float(best_score - baseline_score),
        )

    def _build_decided_preprocessor(self, X: pd.DataFrame) -> ColumnTransformer:
        """Builds a ColumnTransformer honoring Module 2's per-column decisions."""
        decisions = {d.column: d for d in self.plan.column_decisions if d.column in X.columns}

        numeric_cols = [c for c, d in decisions.items() if d.scaling_action == "standard_scale"]
        onehot_cols = [c for c, d in decisions.items() if d.encoding_action == "one_hot"]
        freq_cols = [c for c, d in decisions.items() if d.encoding_action == "frequency_encode"]

        transformers = []
        if numeric_cols:
            numeric_pipe = Pipeline([
                ("impute", SimpleImputer(strategy="median")),
                ("scale", StandardScaler()),
            ])
            transformers.append(("numeric", numeric_pipe, numeric_cols))
        if onehot_cols:
            onehot_pipe = Pipeline([
                ("impute", SimpleImputer(strategy="most_frequent")),
                ("encode", OneHotEncoder(handle_unknown="ignore")),
            ])
            transformers.append(("onehot", onehot_pipe, onehot_cols))
        if freq_cols:
            freq_pipe = Pipeline([
                ("impute", SimpleImputer(strategy="most_frequent")),
                ("encode", FrequencyEncoder()),
            ])
            transformers.append(("freq", freq_pipe, freq_cols))

        return ColumnTransformer(transformers, remainder="drop")

    def _run_baseline(self, X_full: pd.DataFrame, y: pd.Series, metric: str, cv) -> float:
        """Fixed, no-decision pipeline: median-impute + scale numeric, most-frequent
        impute + one-hot everything else, always LogisticRegression. Same treatment
        regardless of dataset shape — this is the thing DataNarrate is supposed to beat."""
        numeric_cols = X_full.select_dtypes(include="number").columns.tolist()
        categorical_cols = X_full.select_dtypes(exclude="number").columns.tolist()

        transformers = []
        if numeric_cols:
            transformers.append(("numeric", Pipeline([
                ("impute", SimpleImputer(strategy="median")),
                ("scale", StandardScaler()),
            ]), numeric_cols))
        if categorical_cols:
            transformers.append(("categorical", Pipeline([
                ("impute", SimpleImputer(strategy="most_frequent")),
                ("encode", OneHotEncoder(handle_unknown="ignore")),
            ]), categorical_cols))

        baseline_preprocessor = ColumnTransformer(transformers, remainder="drop")
        baseline_pipe = Pipeline([
            ("preprocess", baseline_preprocessor),
            ("model", LogisticRegression(max_iter=1000)),
        ])
        scores = cross_val_score(baseline_pipe, X_full, y, cv=cv, scoring=metric)
        return scores.mean()
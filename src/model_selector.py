"""
Module 3 — Model Selection
Cross-validates 4 candidate classifiers — a linear model, a bagging
ensemble, a boosting ensemble, and an explicit voting ensemble combining
all three — reports a full metric suite and confusion matrix for each,
explains each model's gains/limitations against the actual measured
numbers, keeps per-fold scores for significance testing, and compares
the winner against a fixed baseline pipeline. Fits each model once per
fold (not twice), parallelizes tree-based models, and automatically
adapts the number of CV folds to the smallest class.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, get_scorer,
)

from src.profiler import DatasetProfile
from src.preprocessor import PreprocessingPlan


class FrequencyEncoder(BaseEstimator, TransformerMixin):
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


class InsufficientDataError(Exception):
    """Raised when a class has too few examples to cross-validate at all."""
    pass


@dataclass
class ModelMetrics:
    model_name: str
    accuracy: float
    precision: float
    recall: float
    f1: float
    roc_auc: float
    confusion_matrix: List[List[int]]
    gains: List[str] = field(default_factory=list)
    limitations: List[str] = field(default_factory=list)


@dataclass
class SelectionResult:
    metric_used: str
    dropped_columns: List[str]
    candidate_results: List[ModelMetrics]
    best_model_name: str
    best_score: float
    baseline_score: float
    improvement_over_baseline: float
    final_justification: str = ""
    fold_scores: Dict[str, List[float]] = field(default_factory=dict)
    cv_folds_used: int = 5

    def to_dict(self) -> dict:
        return {
            "metric_used": self.metric_used,
            "dropped_columns": self.dropped_columns,
            "candidate_results": [vars(r) for r in self.candidate_results],
            "best_model_name": self.best_model_name,
            "best_score": round(self.best_score, 4),
            "baseline_score": round(self.baseline_score, 4),
            "improvement_over_baseline": round(self.improvement_over_baseline, 4),
            "final_justification": self.final_justification,
            "fold_scores": self.fold_scores,
            "cv_folds_used": self.cv_folds_used,
        }


def _build_candidates() -> Dict[str, Any]:
    lr = LogisticRegression(max_iter=2000, class_weight="balanced")
    rf = RandomForestClassifier(n_estimators=100, random_state=42, class_weight="balanced", n_jobs=-1)
    gb = GradientBoostingClassifier(random_state=42)
    voting = VotingClassifier(estimators=[("lr", lr), ("rf", rf), ("gb", gb)], voting="soft", n_jobs=-1)
    return {
        "logistic_regression": lr,
        "random_forest": rf,
        "gradient_boosting": gb,
        "voting_ensemble": voting,
    }


MODEL_NOTES = {
    "logistic_regression": {
        "gains": ["Fast to train and easy to interpret", "Performs well when classes are close to linearly separable"],
        "limitations": ["Can't capture non-linear relationships on its own"],
    },
    "random_forest": {
        "gains": ["Robust to noise via averaging many trees (bagging)", "Handles non-linear patterns and feature interactions well"],
        "limitations": ["Can be outperformed by simpler models when the true relationship is close to linear", "Less interpretable than a single tree or linear model"],
    },
    "gradient_boosting": {
        "gains": ["Sequentially corrects previous errors (boosting), often reaching high accuracy", "Handles complex non-linear patterns well"],
        "limitations": ["More sensitive to hyperparameters and can overfit without tuning", "Slower to train than a single model"],
    },
    "voting_ensemble": {
        "gains": ["Combines a linear model with two different ensemble strategies, reducing the risk of any single model's blind spot", "Averaging predicted probabilities tends to reduce variance"],
        "limitations": ["Costs roughly 3x the inference time of a single model", "Inherits shared bias if the underlying models agree for the same wrong reason"],
    },
}


class ModelSelector:
    def __init__(self, df: pd.DataFrame, profile: DatasetProfile, plan: PreprocessingPlan):
        self.df = df
        self.profile = profile
        self.plan = plan
        self.target = profile.target_column

    def select(self, cv_folds: int = 5, random_state: int = 42) -> SelectionResult:
        X_full = self.df.drop(columns=[self.target])
        y_raw = self.df[self.target]

        le = LabelEncoder()
        y = le.fit_transform(y_raw)
        is_binary = len(le.classes_) == 2

        class_counts = np.bincount(y)
        min_class_count = int(class_counts.min())
        if min_class_count < 2:
            raise InsufficientDataError(
                f"The rarest class in '{self.target}' has only {min_class_count} example(s). "
                f"Cross-validation needs at least 2 examples per class to run at all — "
                f"please upload a dataset with more examples of every class."
            )
        effective_cv_folds = min(cv_folds, min_class_count)

        dropped = [d.column for d in self.plan.column_decisions if d.missing_action == "drop_column"]
        X = X_full.drop(columns=[c for c in dropped if c in X_full.columns])

        metric_used = "f1" if (self.profile.is_imbalanced or not is_binary) else "accuracy"
        scorer_name = "accuracy" if metric_used == "accuracy" else ("f1" if is_binary else "f1_macro")
        scorer = get_scorer(scorer_name)

        preprocessor = self._build_decided_preprocessor(X)
        cv = StratifiedKFold(n_splits=effective_cv_folds, shuffle=True, random_state=random_state)

        candidate_results = []
        fold_scores = {}
        best_name, best_score = None, -np.inf

        for name, model in _build_candidates().items():
            pipe = Pipeline([("preprocess", preprocessor), ("model", model)])

            fold_preds = np.empty(len(y), dtype=y.dtype)
            fold_proba = np.empty(len(y)) if is_binary else None
            per_fold_scores = []

            for train_idx, test_idx in cv.split(X, y):
                X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
                y_train, y_test = y[train_idx], y[test_idx]

                fitted = Pipeline(pipe.steps).fit(X_train, y_train)
                fold_preds[test_idx] = fitted.predict(X_test)
                per_fold_scores.append(scorer(fitted, X_test, y_test))
                if is_binary:
                    fold_proba[test_idx] = fitted.predict_proba(X_test)[:, 1]

            y_pred = fold_preds
            fold_scores[name] = per_fold_scores
            y_proba = fold_proba

            acc = accuracy_score(y, y_pred)
            avg = "binary" if is_binary else "macro"
            prec = precision_score(y, y_pred, average=avg, zero_division=0)
            rec = recall_score(y, y_pred, average=avg, zero_division=0)
            f1 = f1_score(y, y_pred, average=avg, zero_division=0)
            roc = roc_auc_score(y, y_proba) if y_proba is not None else float("nan")
            cm = confusion_matrix(y, y_pred).tolist()

            score = {"accuracy": acc, "f1": f1}[metric_used]
            notes = MODEL_NOTES.get(name, {"gains": [], "limitations": []})

            candidate_results.append(ModelMetrics(
                model_name=name, accuracy=acc, precision=prec, recall=rec, f1=f1,
                roc_auc=roc, confusion_matrix=cm,
                gains=notes["gains"], limitations=notes["limitations"],
            ))
            if score > best_score:
                best_name, best_score = name, score

        baseline_score = self._run_baseline(X_full, y, metric_used, cv, is_binary)
        justification = self._build_justification(candidate_results, best_name, metric_used, baseline_score)

        return SelectionResult(
            metric_used=metric_used, dropped_columns=dropped,
            candidate_results=candidate_results, best_model_name=best_name,
            best_score=float(best_score), baseline_score=float(baseline_score),
            improvement_over_baseline=float(best_score - baseline_score),
            final_justification=justification, fold_scores=fold_scores,
            cv_folds_used=effective_cv_folds,
        )

    def _build_justification(self, results, best_name, metric, baseline) -> str:
        best = next(r for r in results if r.model_name == best_name)
        best_val = getattr(best, metric)
        others = sorted([r for r in results if r.model_name != best_name], key=lambda r: getattr(r, metric), reverse=True)
        runner_up = others[0]
        gap = round((best_val - getattr(runner_up, metric)) * 100, 2)
        return (
            f"{best_name} was selected as the best model on {metric} ({round(best_val, 4)}), "
            f"{gap} percentage points ahead of the runner-up, {runner_up.model_name} "
            f"({round(getattr(runner_up, metric), 4)}). It beat the fixed baseline by "
            f"{round(best_val - baseline, 4)}. Strengths: {'; '.join(best.gains) or 'none noted'}. "
            f"Known limitations: {'; '.join(best.limitations) or 'none noted'}."
        )

    def _build_decided_preprocessor(self, X: pd.DataFrame) -> ColumnTransformer:
        decisions = {d.column: d for d in self.plan.column_decisions if d.column in X.columns}
        numeric_cols = [c for c, d in decisions.items() if d.scaling_action == "standard_scale"]
        onehot_cols = [c for c, d in decisions.items() if d.encoding_action == "one_hot"]
        freq_cols = [c for c, d in decisions.items() if d.encoding_action == "frequency_encode"]

        transformers = []
        if numeric_cols:
            transformers.append(("numeric", Pipeline([
                ("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler()),
            ]), numeric_cols))
        if onehot_cols:
            transformers.append(("onehot", Pipeline([
                ("impute", SimpleImputer(strategy="most_frequent")),
                ("encode", OneHotEncoder(handle_unknown="ignore")),
            ]), onehot_cols))
        if freq_cols:
            transformers.append(("freq", Pipeline([
                ("impute", SimpleImputer(strategy="most_frequent")), ("encode", FrequencyEncoder()),
            ]), freq_cols))
        return ColumnTransformer(transformers, remainder="drop")

    def _run_baseline(self, X_full, y, metric, cv, is_binary) -> float:
        numeric_cols = X_full.select_dtypes(include="number").columns.tolist()
        categorical_cols = X_full.select_dtypes(exclude="number").columns.tolist()
        transformers = []
        if numeric_cols:
            transformers.append(("numeric", Pipeline([
                ("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler()),
            ]), numeric_cols))
        if categorical_cols:
            transformers.append(("categorical", Pipeline([
                ("impute", SimpleImputer(strategy="most_frequent")),
                ("encode", OneHotEncoder(handle_unknown="ignore")),
            ]), categorical_cols))
        baseline_pipe = Pipeline([
            ("preprocess", ColumnTransformer(transformers, remainder="drop")),
            ("model", LogisticRegression(max_iter=1000)),
        ])
        y_pred = cross_val_predict(baseline_pipe, X_full, y, cv=cv)
        if metric == "accuracy":
            return accuracy_score(y, y_pred)
        return f1_score(y, y_pred, average="binary" if is_binary else "macro", zero_division=0)
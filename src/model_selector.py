"""
Module 3 — Model Selection
Cross-validates 4 candidate classifiers, reports a full metric suite and a
confusion matrix for each, explains each model's gains and limitations, and
compares the winner with a fixed baseline. The baseline is fully described
(what it does, why it was chosen, how it differs from DataNarrate) and an
"improvement ladder" splits the gap into class balancing, preprocessing and
model choice.
"""

from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional

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
class LadderStep:
    label: str
    score: float
    what_changed: str


@dataclass
class BaselineInfo:
    model: str
    metric: str
    score: float
    summary: str
    steps: List[Dict[str, str]]
    why_chosen: List[str]
    caveats: List[str]


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
    baseline_info: Optional[BaselineInfo] = None
    ladder: List[LadderStep] = field(default_factory=list)

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
            "baseline_info": asdict(self.baseline_info) if self.baseline_info is not None else None,
            "ladder": [vars(s) for s in self.ladder],
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


def build_decided_preprocessor(plan: PreprocessingPlan, X: pd.DataFrame) -> ColumnTransformer:
    """Builds the preprocessing exactly as Module 2 decided it, column by column."""
    decisions = {d.column: d for d in plan.column_decisions if d.column in X.columns}
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


def _build_naive_pipeline(X_full: pd.DataFrame, class_weight=None) -> Pipeline:
    """The fixed baseline recipe. It never looks at the data to decide anything."""
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
    return Pipeline([
        ("preprocess", ColumnTransformer(transformers, remainder="drop")),
        ("model", LogisticRegression(max_iter=2000, class_weight=class_weight)),
    ])


def _cv_score(pipe, X, y, cv, metric: str, is_binary: bool) -> float:
    y_pred = cross_val_predict(pipe, X, y, cv=cv)
    if metric == "accuracy":
        return float(accuracy_score(y, y_pred))
    return float(f1_score(y, y_pred, average="binary" if is_binary else "macro", zero_division=0))


def _names(cols: List[str], limit: int = 6) -> str:
    if len(cols) <= limit:
        return ", ".join(cols)
    return ", ".join(cols[:limit]) + f" (+{len(cols) - limit} more)"


def _describe_baseline(plan: PreprocessingPlan, metric_used: str, score: float) -> BaselineInfo:
    decisions = plan.column_decisions
    ids = [d.column for d in decisions if any("identifier" in r for r in d.reasons)]
    heavy_missing = [d.column for d in decisions if d.missing_action == "drop_column" and d.column not in ids]
    num_imputed = [d.column for d in decisions if d.missing_action == "impute_median"]
    cat_imputed = [d.column for d in decisions if d.missing_action == "impute_mode"]
    onehot = [d.column for d in decisions if d.encoding_action == "one_hot"]
    freq = [d.column for d in decisions if d.encoding_action == "frequency_encode"]

    base_key = "Baseline (fixed recipe)"
    dn_key = "DataNarrate (decided from the data)"

    steps = [
        {"Stage": "Identifier-like columns",
         base_key: "Kept, and one-hot encoded like any other text column",
         dn_key: f"Dropped: {_names(ids)}" if ids else "None found, nothing dropped"},
        {"Stage": "Columns with 40% or more missing",
         base_key: "Kept and filled in",
         dn_key: f"Dropped: {_names(heavy_missing)}" if heavy_missing else "None found, nothing dropped"},
        {"Stage": "Missing numbers (under 40%)",
         base_key: "Filled with the column median",
         dn_key: f"Same: median ({len(num_imputed)} column(s) needed it)"},
        {"Stage": "Missing categories (under 40%)",
         base_key: "Filled with the most frequent category",
         dn_key: f"Same: most frequent category ({len(cat_imputed)} column(s) needed it)"},
        {"Stage": "Categorical encoding",
         base_key: "One-hot encode every categorical column, however many categories it has",
         dn_key: f"One-hot for {len(onehot)} column(s)"
                 + (f"; frequency encoding for {len(freq)} high-cardinality column(s): {_names(freq)}" if freq else "")},
        {"Stage": "Numeric scaling",
         base_key: "Standardize every numeric column",
         dn_key: "Same: standardize every numeric column"},
        {"Stage": "Class imbalance",
         base_key: "Ignored: every row counts equally",
         dn_key: "Logistic regression and random forest are trained with balanced class weights"},
        {"Stage": "Model",
         base_key: "Logistic regression only",
         dn_key: "Compares 4 models (logistic regression, random forest, gradient boosting, voting ensemble) and keeps the best"},
    ]

    summary = (
        "Logistic regression with default settings and no class weighting, trained on a fixed preprocessing "
        "recipe that never looks at the data: numeric columns are filled with the median and standardized, "
        "every categorical column is filled with its most frequent value and one-hot encoded, and no column "
        f"is ever removed. It is scored with the same cross-validation folds and the same metric ({metric_used}) "
        "as every candidate model."
    )

    why_chosen = [
        "It represents what a practitioner would do with no analysis: apply the same standard recipe to any dataset. If DataNarrate cannot beat that, the analysis added nothing.",
        "Logistic regression is the usual first model for classification. It is fast, deterministic and widely used as a reference point.",
        "It is not a strawman. It has sensible defaults (it fills missing values, scales numbers and encodes text), so it never crashes on messy data.",
        "It uses exactly the same rows, cross-validation folds and metric as every candidate, so the comparison is like for like.",
    ]

    caveats = [
        "The baseline and DataNarrate differ in several ways at once, so the gap is a bundle of effects. "
        "The improvement ladder separates them one change at a time.",
    ]
    if ids:
        caveats.append(
            f"The baseline one-hot encodes identifier-like columns ({_names(ids)}), which gives it many useless "
            f"features. Part of the gap comes from DataNarrate removing them."
        )

    return BaselineInfo(
        model="logistic_regression", metric=metric_used, score=float(score),
        summary=summary, steps=steps, why_chosen=why_chosen, caveats=caveats,
    )


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

        preprocessor = build_decided_preprocessor(self.plan, X)
        cv = StratifiedKFold(n_splits=effective_cv_folds, shuffle=True, random_state=random_state)

        candidate_results = []
        fold_scores = {}
        model_scores = {}
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
            model_scores[name] = float(score)
            notes = MODEL_NOTES.get(name, {"gains": [], "limitations": []})

            candidate_results.append(ModelMetrics(
                model_name=name, accuracy=acc, precision=prec, recall=rec, f1=f1,
                roc_auc=roc, confusion_matrix=cm,
                gains=notes["gains"], limitations=notes["limitations"],
            ))
            if score > best_score:
                best_name, best_score = name, score

        baseline_score = _cv_score(_build_naive_pipeline(X_full, None), X_full, y, cv, metric_used, is_binary)
        balanced_baseline_score = _cv_score(
            _build_naive_pipeline(X_full, "balanced"), X_full, y, cv, metric_used, is_binary
        )

        best_is_lr = best_name == "logistic_regression"
        ladder = [
            LadderStep("Fixed baseline", baseline_score,
                       "Logistic regression on the fixed preprocessing recipe"),
            LadderStep("+ class balancing", balanced_baseline_score,
                       "Same recipe and model, but the minority class is weighted up so it counts more"),
            LadderStep("+ DataNarrate preprocessing", model_scores["logistic_regression"],
                       "Same model, but preprocessing is decided from the data instead of the fixed recipe"),
            LadderStep("+ best model chosen", float(best_score),
                       "Logistic regression was already the best model, so nothing changes" if best_is_lr
                       else f"Switched from logistic regression to {best_name}"),
        ]

        baseline_info = _describe_baseline(self.plan, metric_used, baseline_score)
        justification = self._build_justification(candidate_results, best_name, metric_used, baseline_score)

        return SelectionResult(
            metric_used=metric_used, dropped_columns=dropped,
            candidate_results=candidate_results, best_model_name=best_name,
            best_score=float(best_score), baseline_score=float(baseline_score),
            improvement_over_baseline=float(best_score - baseline_score),
            final_justification=justification, fold_scores=fold_scores,
            cv_folds_used=effective_cv_folds, baseline_info=baseline_info, ladder=ladder,
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
            f"({round(getattr(runner_up, metric), 4)}). Compared with the fixed baseline "
            f"(logistic regression on a fixed preprocessing recipe, {round(baseline, 4)}), the score changed by "
            f"{best_val - baseline:+.4f}. Strengths: {'; '.join(best.gains) or 'none noted'}. "
            f"Known limitations: {'; '.join(best.limitations) or 'none noted'}."
        )
"""
Ablation Study — measures how much each preprocessing decision actually
contributes, by disabling one step at a time and re-scoring the winning
model. This is what proves the decisions matter individually, not just
that the model choice matters.
"""
from dataclasses import dataclass
from typing import List
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import f1_score, accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier

from src.profiler import DatasetProfile
from src.preprocessor import PreprocessingPlan
from src.model_selector import FrequencyEncoder


@dataclass
class AblationResult:
    variant_name: str
    score: float
    score_drop_from_full: float


MODEL_BUILDERS = {
    "logistic_regression": lambda: LogisticRegression(max_iter=1000),
    "random_forest": lambda: RandomForestClassifier(n_estimators=200, random_state=42),
    "gradient_boosting": lambda: GradientBoostingClassifier(random_state=42),
    "voting_ensemble": lambda: VotingClassifier(estimators=[
        ("lr", LogisticRegression(max_iter=1000)),
        ("rf", RandomForestClassifier(n_estimators=200, random_state=42)),
        ("gb", GradientBoostingClassifier(random_state=42)),
    ], voting="soft"),
}


def run_ablation(df: pd.DataFrame, profile: DatasetProfile, plan: PreprocessingPlan,
                  best_model_name: str, metric: str, cv_folds: int = 5) -> List[AblationResult]:
    X_full = df.drop(columns=[profile.target_column])
    y = LabelEncoder().fit_transform(df[profile.target_column])
    dropped = [d.column for d in plan.column_decisions if d.missing_action == "drop_column"]
    X = X_full.drop(columns=[c for c in dropped if c in X_full.columns])
    decisions = {d.column: d for d in plan.column_decisions if d.column in X.columns}
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
    is_binary = len(set(y)) == 2

    def score_of(pipe) -> float:
        y_pred = cross_val_predict(pipe, X, y, cv=cv)
        if metric == "accuracy":
            return accuracy_score(y, y_pred)
        return f1_score(y, y_pred, average="binary" if is_binary else "macro", zero_division=0)

    def build_preprocessor(use_scaling=True, use_smart_encoding=True, use_imputation=True) -> ColumnTransformer:
        numeric_cols = [c for c, d in decisions.items() if d.scaling_action == "standard_scale"]
        onehot_cols = [c for c, d in decisions.items() if d.encoding_action == "one_hot"]
        freq_cols = [c for c, d in decisions.items() if d.encoding_action == "frequency_encode"]

        impute_strategy_num = "median" if use_imputation else "constant"
        impute_strategy_cat = "most_frequent" if use_imputation else "constant"
        num_fill = {} if use_imputation else {"fill_value": 0}
        cat_fill = {} if use_imputation else {"fill_value": "missing"}

        transformers = []
        if numeric_cols:
            steps = [("impute", SimpleImputer(strategy=impute_strategy_num, **num_fill))]
            if use_scaling:
                steps.append(("scale", StandardScaler()))
            transformers.append(("numeric", Pipeline(steps), numeric_cols))
        if onehot_cols or freq_cols:
            all_cat = onehot_cols + freq_cols
            encoder = OneHotEncoder(handle_unknown="ignore") if not use_smart_encoding else None
            if use_smart_encoding:
                if onehot_cols:
                    transformers.append(("onehot", Pipeline([
                        ("impute", SimpleImputer(strategy=impute_strategy_cat, **cat_fill)),
                        ("encode", OneHotEncoder(handle_unknown="ignore")),
                    ]), onehot_cols))
                if freq_cols:
                    transformers.append(("freq", Pipeline([
                        ("impute", SimpleImputer(strategy=impute_strategy_cat, **cat_fill)),
                        ("encode", FrequencyEncoder()),
                    ]), freq_cols))
            else:
                transformers.append(("cat_forced_onehot", Pipeline([
                    ("impute", SimpleImputer(strategy=impute_strategy_cat, **cat_fill)),
                    ("encode", OneHotEncoder(handle_unknown="ignore")),
                ]), all_cat))
        return ColumnTransformer(transformers, remainder="drop")

    model_fn = MODEL_BUILDERS[best_model_name]
    variants = {
        "full_pipeline": build_preprocessor(True, True, True),
        "no_scaling": build_preprocessor(False, True, True),
        "no_smart_encoding_forced_onehot": build_preprocessor(True, False, True),
        "no_imputation_zero_fill": build_preprocessor(True, True, False),
    }

    results = []
    full_score = None
    for name, preprocessor in variants.items():
        pipe = Pipeline([("preprocess", preprocessor), ("model", model_fn())])
        score = score_of(pipe)
        if name == "full_pipeline":
            full_score = score
        results.append(AblationResult(variant_name=name, score=score, score_drop_from_full=0.0))

    for r in results:
        r.score_drop_from_full = round(full_score - r.score, 4)
    return results
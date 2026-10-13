"""
Preprocessing Proof
For every column, re-runs the decided preprocessing steps on the real data
and records evidence that each step did what it claimed: before/after
statistics, pass/fail checks and example rows. Values shown here are computed
on the full data for display; during model evaluation the same steps are fit
inside each cross-validation fold.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

from src.profiler import DatasetProfile
from src.preprocessor import (
    PreprocessingPlan, ColumnDecision, MISSING_DROP_THRESHOLD, ID_LIKE_UNIQUE_RATIO,
)
from src.model_selector import FrequencyEncoder, build_decided_preprocessor

PLACEHOLDER_TOKENS = {"?", "NA", "N/A", "n/a", "null", "NULL", "None", "nan", "NaN", "-", "--", ""}


@dataclass
class ColumnProof:
    column: str
    actions: List[str]
    reasons: List[str]
    before: Dict[str, Any]
    after: Dict[str, Any]
    checks: List[Tuple[str, bool, str]]
    examples: List[Dict[str, str]]
    warnings: List[str] = field(default_factory=list)

    @property
    def all_passed(self) -> bool:
        return all(ok for _, ok, _ in self.checks)


@dataclass
class PreprocessingProof:
    columns: List[ColumnProof]
    n_rows: int
    n_features_before: int
    n_features_after: int
    nan_cells_after: int
    full_check_passed: bool
    notes: List[str]


def _fmt(x: Any) -> Any:
    if isinstance(x, (float, np.floating)):
        return round(float(x), 4)
    if isinstance(x, np.integer):
        return int(x)
    return x


def _s(x: Any) -> str:
    return str(_fmt(x))


def _actions_text(d: ColumnDecision) -> List[str]:
    acts = []
    if d.missing_action == "drop_column":
        acts.append("drop column")
    if d.missing_action == "impute_median":
        acts.append("fill missing with median")
    if d.missing_action == "impute_mode":
        acts.append("fill missing with most frequent category")
    if d.encoding_action == "one_hot":
        acts.append("one-hot encode")
    if d.encoding_action == "frequency_encode":
        acts.append("frequency encode")
    if d.scaling_action == "standard_scale":
        acts.append("standardize")
    return acts or ["no change"]


def _example_positions(missing_mask: np.ndarray, limit: int = 5) -> List[int]:
    missing_pos = [int(i) for i in np.where(missing_mask)[0][:3]]
    others = [i for i in range(min(len(missing_mask), 10)) if i not in missing_pos]
    return missing_pos + others[: limit - len(missing_pos)]


def _placeholder_warnings(series: pd.Series) -> List[str]:
    text = series.dropna().astype(str).str.strip()
    hits = text[text.isin(PLACEHOLDER_TOKENS)]
    return [
        f"{int(n)} row(s) contain '{tok}', which looks like a missing-value placeholder. "
        f"It is being treated as an ordinary category, not as missing data."
        for tok, n in hits.value_counts().items()
    ]


def _prove_numeric(d: ColumnDecision, series: pd.Series) -> ColumnProof:
    raw = pd.to_numeric(series, errors="coerce")
    n_missing = int(raw.isna().sum())
    if raw.notna().sum() == 0:
        return ColumnProof(
            column=d.column, actions=_actions_text(d), reasons=d.reasons,
            before={"missing cells": n_missing}, after={},
            checks=[("Column has numeric values", False, "every value is missing or non-numeric")],
            examples=[],
        )

    before = {
        "missing cells": n_missing, "mean": _fmt(raw.mean()), "std": _fmt(raw.std(ddof=0)),
        "min": _fmt(raw.min()), "max": _fmt(raw.max()),
    }

    imputer = SimpleImputer(strategy="median")
    imputed = imputer.fit_transform(raw.to_frame())[:, 0]
    fill = float(imputer.statistics_[0])
    scaled = StandardScaler().fit_transform(imputed.reshape(-1, 1))[:, 0]

    after = {
        "missing cells": int(np.isnan(scaled).sum()), "mean": _fmt(scaled.mean()),
        "std": _fmt(scaled.std()), "min": _fmt(scaled.min()), "max": _fmt(scaled.max()),
    }

    checks: List[Tuple[str, bool, str]] = []
    if n_missing:
        checks.append((
            "Missing cells filled", int(np.isnan(imputed).sum()) == 0,
            f"{n_missing} cell(s) filled with the median, {fill:.4g}",
        ))
    else:
        checks.append(("No missing cells", True, "nothing needed filling"))

    constant = float(np.std(imputed)) == 0.0
    mean_ok = abs(float(scaled.mean())) < 1e-6
    std_ok = abs(float(scaled.std()) - 1.0) < 1e-6 or constant
    checks.append((
        "Scaled to mean 0 and standard deviation 1", mean_ok and std_ok,
        f"after scaling: mean={scaled.mean():.2e}, std={scaled.std():.4f}"
        + (" (column is constant)" if constant else ""),
    ))

    examples = []
    for i in _example_positions(raw.isna().to_numpy()):
        examples.append({
            "row": str(series.index[i]),
            "raw value": "(missing)" if pd.isna(raw.iloc[i]) else _s(raw.iloc[i]),
            "after filling": _s(imputed[i]),
            "final (standardized)": _s(scaled[i]),
        })

    return ColumnProof(
        column=d.column, actions=_actions_text(d), reasons=d.reasons,
        before=before, after=after, checks=checks, examples=examples,
    )


def _prove_categorical(d: ColumnDecision, series: pd.Series) -> ColumnProof:
    n_missing = int(series.isna().sum())
    counts = series.value_counts(dropna=True)
    if len(counts) == 0:
        return ColumnProof(
            column=d.column, actions=_actions_text(d), reasons=d.reasons,
            before={"missing cells": n_missing}, after={},
            checks=[("Column has values", False, "every value is missing")], examples=[],
        )

    top_share = counts.iloc[0] / counts.sum()
    before = {
        "missing cells": n_missing, "distinct categories": int(len(counts)),
        "most common": f"{counts.index[0]} ({top_share:.1%})",
    }

    imputer = SimpleImputer(strategy="most_frequent")
    filled = imputer.fit_transform(series.to_frame())[:, 0]
    fill = imputer.statistics_[0]
    filled_df = pd.DataFrame(filled)
    n_left = int(pd.isna(filled).sum())

    checks: List[Tuple[str, bool, str]] = []
    if n_missing:
        checks.append((
            "Missing cells filled", n_left == 0,
            f"{n_missing} cell(s) filled with the most frequent category, '{fill}'",
        ))
    else:
        checks.append(("No missing cells", True, "nothing needed filling"))

    positions = _example_positions(series.isna().to_numpy())
    after: Dict[str, Any] = {"missing cells": n_left}
    final_values: Dict[int, str] = {i: "-" for i in positions}

    if d.encoding_action == "one_hot":
        enc = OneHotEncoder(handle_unknown="ignore")
        matrix = enc.fit_transform(filled_df)
        n_out = int(matrix.shape[1])
        row_sums = np.asarray(matrix.sum(axis=1)).ravel()
        n_distinct = int(pd.Series(filled).nunique())
        after["output columns"] = n_out
        checks.append((
            "Every row has exactly one 1", bool(np.all(row_sums == 1)),
            f"row sums: min={row_sums.min():.0f}, max={row_sums.max():.0f}",
        ))
        checks.append((
            "One new column per category", n_out == n_distinct,
            f"{n_distinct} categories became {n_out} columns",
        ))
        active = np.asarray(matrix.argmax(axis=1)).ravel()
        cats = enc.categories_[0]
        final_values = {i: f"{d.column}_{cats[active[i]]} = 1" for i in positions}

    elif d.encoding_action == "frequency_encode":
        fe = FrequencyEncoder().fit(filled_df)
        vals = fe.transform(filled_df)[:, 0]
        fmap = fe.freq_maps_[0]
        top3 = sorted(fmap.items(), key=lambda kv: kv[1], reverse=True)[:3]
        after["smallest frequency"] = _fmt(vals.min())
        after["largest frequency"] = _fmt(vals.max())
        after["distinct output values"] = int(len(np.unique(vals)))
        after["most common categories"] = ", ".join(f"{k} = {v:.1%}" for k, v in top3)
        checks.append((
            "Every value is a frequency between 0 and 1",
            bool(np.all((vals >= 0) & (vals <= 1))),
            f"range {vals.min():.4f} to {vals.max():.4f}",
        ))
        checks.append((
            "Frequencies add up to 100%", abs(sum(fmap.values()) - 1.0) < 1e-9,
            f"sum = {sum(fmap.values()):.6f}",
        ))
        final_values = {i: f"{vals[i]:.4f}" for i in positions}

    examples = []
    for i in positions:
        examples.append({
            "row": str(series.index[i]),
            "raw value": "(missing)" if pd.isna(series.iloc[i]) else str(series.iloc[i]),
            "after filling": str(filled[i]),
            "final": final_values[i],
        })

    return ColumnProof(
        column=d.column, actions=_actions_text(d), reasons=d.reasons,
        before=before, after=after, checks=checks, examples=examples,
        warnings=_placeholder_warnings(series),
    )


def _prove_dropped(d: ColumnDecision, series: pd.Series, n_rows: int) -> ColumnProof:
    n_unique = int(series.nunique(dropna=True))
    ratio = n_unique / n_rows if n_rows else 0.0
    miss_pct = float(series.isna().mean() * 100)
    is_categorical = not pd.api.types.is_numeric_dtype(series)
    id_rule = is_categorical and ratio >= ID_LIKE_UNIQUE_RATIO
    missing_rule = miss_pct >= MISSING_DROP_THRESHOLD

    before = {
        "missing cells": int(series.isna().sum()), "missing %": f"{miss_pct:.2f}%",
        "distinct values": n_unique, "distinct values / rows": f"{ratio:.1%}",
    }

    if id_rule:
        check = ("Identifier rule met", True,
                 f"{n_unique} distinct values across {n_rows} rows = {ratio:.1%} "
                 f"(rule: at least {ID_LIKE_UNIQUE_RATIO:.0%})")
    elif missing_rule:
        check = ("Missing-data rule met", True,
                 f"{miss_pct:.2f}% missing (rule: at least {MISSING_DROP_THRESHOLD:.0f}%)")
    else:
        check = ("Drop rule met", False,
                 "neither the identifier rule nor the missing-data rule applies to this data")

    examples = [
        {"row": str(series.index[i]), "raw value": str(series.iloc[i])}
        for i in range(min(3, len(series)))
    ]
    return ColumnProof(
        column=d.column, actions=_actions_text(d), reasons=d.reasons,
        before=before, after={"status": "removed from the model's input"},
        checks=[check], examples=examples,
    )


def build_preprocessing_proof(
    df: pd.DataFrame, profile: DatasetProfile, plan: PreprocessingPlan
) -> PreprocessingProof:
    n_rows = len(df)
    proofs: List[ColumnProof] = []

    for d in plan.column_decisions:
        series = df[d.column]
        if d.missing_action == "drop_column":
            proof = _prove_dropped(d, series, n_rows)
        elif pd.api.types.is_numeric_dtype(series) and d.scaling_action == "standard_scale":
            proof = _prove_numeric(d, series)
        elif d.encoding_action in ("one_hot", "frequency_encode"):
            proof = _prove_categorical(d, series)
        else:
            proof = ColumnProof(
                column=d.column, actions=_actions_text(d), reasons=d.reasons,
                before={}, after={}, checks=[("No action needed", True, "")], examples=[],
            )
        proofs.append(proof)

    dropped = [d.column for d in plan.column_decisions if d.missing_action == "drop_column"]
    X = df.drop(columns=[profile.target_column] + dropped, errors="ignore")
    transformed = build_decided_preprocessor(plan, X).fit_transform(X)
    if sparse.issparse(transformed):
        nan_cells = int(np.isnan(transformed.data).sum())
    else:
        nan_cells = int(np.isnan(np.asarray(transformed, dtype=float)).sum())

    return PreprocessingProof(
        columns=proofs,
        n_rows=n_rows,
        n_features_before=int(X.shape[1]),
        n_features_after=int(transformed.shape[1]),
        nan_cells_after=nan_cells,
        full_check_passed=nan_cells == 0 and all(p.all_passed for p in proofs),
        notes=[
            "Values here are computed by fitting each step on the full data, for display only. "
            "During model evaluation the same steps are fit separately inside each cross-validation "
            "fold, so rows used for testing never influence the preprocessing."
        ],
    )
"""
Outlier handling (rule R8).
Outliers are found with the IQR rule: anything below Q1 - 1.5*IQR or above
Q3 + 1.5*IQR is an outlier. When the IQR is 0 (e.g. a column that is mostly
zeros) the fences are useless, so the 1st and 99th percentiles are used as
the clip bounds instead.

Rows are never deleted — extreme values are clipped to the bounds. The bounds
are learned in fit() from the training data only, so inside cross-validation
every fold learns its own bounds and nothing leaks from the test fold.
"""

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin


IQR_FACTOR = 1.5
LOW_PERCENTILE = 1
HIGH_PERCENTILE = 99


def iqr_fences(values):
    """Returns (q1, q3, iqr, lower_fence, upper_fence) for an array of numbers."""
    q1, q3 = np.percentile(values, [25, 75])
    iqr = q3 - q1
    return q1, q3, iqr, q1 - IQR_FACTOR * iqr, q3 + IQR_FACTOR * iqr


class OutlierClipper(BaseEstimator, TransformerMixin):
    """
    Clips each column to bounds learned from the data passed to fit().
    methods has one entry per column: "iqr", "percentile" or "none".
    """

    def __init__(self, methods=None):
        self.methods = methods

    def fit(self, X, y=None):
        X = np.asarray(X, dtype=float)
        n_cols = X.shape[1]
        methods = self.methods if self.methods is not None else ["iqr"] * n_cols

        self.lower_ = np.full(n_cols, -np.inf)
        self.upper_ = np.full(n_cols, np.inf)

        for j, method in enumerate(methods):
            values = X[:, j]
            values = values[~np.isnan(values)]
            if method == "none" or len(values) == 0:
                continue
            if method == "iqr":
                _, _, iqr, lower, upper = iqr_fences(values)
                if iqr == 0:
                    continue  # fences would be useless, leave the column alone
                self.lower_[j], self.upper_[j] = lower, upper
            elif method == "percentile":
                self.lower_[j], self.upper_[j] = np.percentile(values, [LOW_PERCENTILE, HIGH_PERCENTILE])
            else:
                raise ValueError(f"Unknown clipping method: {method}")
        return self

    def transform(self, X):
        X = np.asarray(X, dtype=float)
        return np.clip(X, self.lower_, self.upper_)  # NaN stays NaN


def build_clipper(decisions, numeric_cols):
    """
    Makes an OutlierClipper for the given numeric columns from the plan's R8
    decisions (decisions maps column name -> ColumnDecision).
    Returns None when no column needs clipping.
    """
    methods = []
    for col in numeric_cols:
        d = decisions[col]
        methods.append(d.outlier_bounds if d.outlier_action == "clip" else "none")
    if all(m == "none" for m in methods):
        return None
    return OutlierClipper(methods=methods)


def outlier_rows(profile, plan):
    """One row per kept numeric column, for showing the R8 decisions in a table."""
    decisions = {d.column: d for d in plan.column_decisions}
    rows = []
    for col in profile.columns:
        d = decisions[col.name]
        if col.dtype != "numeric" or d.missing_action == "drop_column":
            continue
        reason = next((r for r in d.reasons if r.startswith("R8")), "")
        rows.append({
            "column": col.name,
            "outlier_pct": col.outlier_pct,
            "lower_fence": col.lower_fence,
            "upper_fence": col.upper_fence,
            "action": d.outlier_action,
            "clip_lower": d.clip_lower,
            "clip_upper": d.clip_upper,
            "reason": reason,
        })
    return rows

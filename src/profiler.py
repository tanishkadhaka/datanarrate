"""
Module 1 — Data Profiling
Reads a dataset and computes a structured profile of its properties.
This profile is the single input every later decision (Module 2, Module 3) reads from.
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional

from src.outlier import iqr_fences, LOW_PERCENTILE, HIGH_PERCENTILE


@dataclass
class ColumnProfile:
    name: str
    dtype: str
    missing_pct: float
    n_unique: int
    is_high_cardinality: bool
    # Outlier stats (numeric columns only). Counted among the non-missing values.
    outlier_count: int = 0
    outlier_pct: float = 0.0
    lower_fence: Optional[float] = None
    upper_fence: Optional[float] = None
    iqr_is_zero: bool = False
    p01: Optional[float] = None   # 1st percentile, clip bound when the IQR is 0
    p99: Optional[float] = None   # 99th percentile


@dataclass
class DatasetProfile:
    n_rows: int
    n_cols: int
    target_column: str
    task_type: str  # locked to "classification" for this project's scope
    n_classes: int
    class_distribution: Dict[Any, float]
    is_imbalanced: bool
    columns: List[ColumnProfile] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "n_rows": self.n_rows,
            "n_cols": self.n_cols,
            "target_column": self.target_column,
            "task_type": self.task_type,
            "n_classes": self.n_classes,
            "class_distribution": self.class_distribution,
            "is_imbalanced": self.is_imbalanced,
            "columns": [vars(c) for c in self.columns],
        }


HIGH_CARDINALITY_THRESHOLD = 20   # unique values in a categorical column
IMBALANCE_RATIO_THRESHOLD = 0.4   # smallest_class / largest_class below this = imbalanced


class DataProfiler:
    """Profiles a pandas DataFrame for a classification task."""

    def __init__(self, df: pd.DataFrame, target_column: str):
        if target_column not in df.columns:
            raise ValueError(f"Target column '{target_column}' not found in dataset.")
        self.df = df
        self.target_column = target_column

    def profile(self) -> DatasetProfile:
        df = self.df
        target = self.target_column

        n_rows, n_cols = df.shape

        columns = []
        for col in df.columns:
            if col == target:
                continue
            missing_pct = round(df[col].isna().mean() * 100, 2)
            n_unique = df[col].nunique(dropna=True)
            dtype = self._infer_dtype(df[col])
            is_high_card = dtype == "categorical" and n_unique > HIGH_CARDINALITY_THRESHOLD
            outlier_stats = self._outlier_stats(df[col]) if dtype == "numeric" else {}
            columns.append(ColumnProfile(
                name=col,
                dtype=dtype,
                missing_pct=missing_pct,
                n_unique=n_unique,
                is_high_cardinality=is_high_card,
                **outlier_stats,
            ))

        class_counts = df[target].value_counts(dropna=True)
        n_classes = class_counts.shape[0]
        class_distribution = (class_counts / class_counts.sum()).round(4).to_dict()
        is_imbalanced = self._check_imbalance(class_counts)

        return DatasetProfile(
            n_rows=n_rows,
            n_cols=n_cols,
            target_column=target,
            task_type="classification",
            n_classes=n_classes,
            class_distribution=class_distribution,
            is_imbalanced=is_imbalanced,
            columns=columns,
        )

    @staticmethod
    def _infer_dtype(series: pd.Series) -> str:
        if pd.api.types.is_numeric_dtype(series):
            return "numeric"
        return "categorical"

    @staticmethod
    def _outlier_stats(series: pd.Series) -> dict:
        """IQR outlier stats for one numeric column (missing and infinite values ignored)."""
        values = series.dropna().astype(float).to_numpy()
        values = values[np.isfinite(values)]
        if len(values) == 0:
            return {}

        _, _, iqr, lower, upper = iqr_fences(values)
        n_outliers = int(((values < lower) | (values > upper)).sum())
        p01, p99 = np.percentile(values, [LOW_PERCENTILE, HIGH_PERCENTILE])
        return {
            "outlier_count": n_outliers,
            "outlier_pct": round(n_outliers / len(values) * 100, 2),
            "lower_fence": float(lower),
            "upper_fence": float(upper),
            "iqr_is_zero": bool(iqr == 0),
            "p01": float(p01),
            "p99": float(p99),
        }

    @staticmethod
    def _check_imbalance(class_counts: pd.Series) -> bool:
        if len(class_counts) < 2:
            return False
        ratio = class_counts.min() / class_counts.max()
        return bool(ratio < IMBALANCE_RATIO_THRESHOLD)
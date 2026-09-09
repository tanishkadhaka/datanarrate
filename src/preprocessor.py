"""
Module 2 — Preprocessing Decider
Reads a DatasetProfile (from Module 1) and decides, column by column, how to
handle missing values, encoding, and scaling — using explicit rules, not a
trained model. Every decision carries a plain-language reason.
"""

from dataclasses import dataclass, field
from typing import List
from src.profiler import DatasetProfile, ColumnProfile


MISSING_DROP_THRESHOLD = 40.0  # % missing above which we drop the column


@dataclass
class ColumnDecision:
    column: str
    missing_action: str        # "none" | "impute_median" | "impute_mode" | "drop_column"
    encoding_action: str       # "none" | "one_hot" | "frequency_encode"
    scaling_action: str        # "none" | "standard_scale"
    reasons: List[str] = field(default_factory=list)


@dataclass
class PreprocessingPlan:
    column_decisions: List[ColumnDecision]
    summary: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "column_decisions": [vars(c) for c in self.column_decisions],
            "summary": self.summary,
        }


class PreprocessingDecider:
    """Applies explicit, hand-written rules to a DatasetProfile."""

    def __init__(self, profile: DatasetProfile):
        self.profile = profile

    def decide(self) -> PreprocessingPlan:
        decisions = [self._decide_column(col) for col in self.profile.columns]
        summary = self._build_summary(decisions)
        return PreprocessingPlan(column_decisions=decisions, summary=summary)

    def _decide_column(self, col: ColumnProfile) -> ColumnDecision:
        reasons = []

        # --- Missing value handling ---
        if col.missing_pct >= MISSING_DROP_THRESHOLD:
            missing_action = "drop_column"
            reasons.append(
                f"{col.missing_pct}% missing exceeds the {MISSING_DROP_THRESHOLD}% "
                f"threshold — imputing this much data would be unreliable, so the "
                f"column is dropped."
            )
        elif col.missing_pct > 0:
            if col.dtype == "numeric":
                missing_action = "impute_median"
                reasons.append(
                    f"{col.missing_pct}% missing — median imputation preserves the "
                    f"row and is robust to outliers, unlike mean imputation."
                )
            else:
                missing_action = "impute_mode"
                reasons.append(
                    f"{col.missing_pct}% missing — imputing with the most frequent "
                    f"category keeps the row without inventing a numeric value."
                )
        else:
            missing_action = "none"

        # --- Encoding (categorical only) ---
        encoding_action = "none"
        if col.dtype == "categorical" and missing_action != "drop_column":
            if col.is_high_cardinality:
                encoding_action = "frequency_encode"
                reasons.append(
                    f"{col.n_unique} unique categories is high-cardinality — one-hot "
                    f"encoding would explode the feature space, so frequency "
                    f"encoding is used instead."
                )
            else:
                encoding_action = "one_hot"
                reasons.append(
                    f"{col.n_unique} unique categories is low enough for one-hot "
                    f"encoding without blowing up dimensionality."
                )

        # --- Scaling (numeric only) ---
        scaling_action = "none"
        if col.dtype == "numeric" and missing_action != "drop_column":
            scaling_action = "standard_scale"
            reasons.append(
                "Numeric column — standardized so distance- and gradient-based "
                "models (e.g. logistic regression, SVM, KNN) aren't dominated by "
                "columns with larger raw scales."
            )

        return ColumnDecision(
            column=col.name,
            missing_action=missing_action,
            encoding_action=encoding_action,
            scaling_action=scaling_action,
            reasons=reasons,
        )

    def _build_summary(self, decisions: List[ColumnDecision]) -> List[str]:
        dropped = [d.column for d in decisions if d.missing_action == "drop_column"]
        imputed = [d.column for d in decisions if d.missing_action.startswith("impute")]
        encoded = [d.column for d in decisions if d.encoding_action != "none"]
        scaled = [d.column for d in decisions if d.scaling_action != "none"]

        summary = []
        if dropped:
            summary.append(f"{len(dropped)} column(s) dropped for excessive missingness: {dropped}")
        if imputed:
            summary.append(f"{len(imputed)} column(s) imputed: {imputed}")
        if encoded:
            summary.append(f"{len(encoded)} categorical column(s) encoded: {encoded}")
        if scaled:
            summary.append(f"{len(scaled)} numeric column(s) scheduled for scaling")
        if not summary:
            summary.append("No preprocessing needed — dataset is already clean.")
        return summary
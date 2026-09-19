"""
Module 2 — Preprocessing Decider
Reads a DatasetProfile and decides, column by column, how to handle missing
values, encoding, and scaling — using explicit rules, not a trained model.
Every decision carries a plain-language reason, and the full sequence of
actions is recorded as a readable step trace.
"""

from dataclasses import dataclass, field
from typing import List
from src.profiler import DatasetProfile, ColumnProfile


MISSING_DROP_THRESHOLD = 40.0   # % missing above which we drop the column
ID_LIKE_UNIQUE_RATIO = 0.95     # unique/total rows above this = identifier, not a feature


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
    step_trace: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "column_decisions": [vars(c) for c in self.column_decisions],
            "summary": self.summary,
            "step_trace": self.step_trace,
        }


class PreprocessingDecider:
    def __init__(self, profile: DatasetProfile):
        self.profile = profile

    def decide(self) -> PreprocessingPlan:
        decisions = [self._decide_column(col) for col in self.profile.columns]
        summary = self._build_summary(decisions)
        step_trace = self._build_step_trace(decisions)
        return PreprocessingPlan(column_decisions=decisions, summary=summary, step_trace=step_trace)

    def _decide_column(self, col: ColumnProfile) -> ColumnDecision:
        reasons = []
        n_rows = self.profile.n_rows

        # --- Identifier-column check (runs first, overrides everything else) ---
        if col.dtype == "categorical" and n_rows > 0 and (col.n_unique / n_rows) >= ID_LIKE_UNIQUE_RATIO:
            reasons.append(
                f"{col.n_unique} unique values across {n_rows} rows means this is almost "
                f"certainly an identifier column (e.g. a customer ID), not a real feature — "
                f"encoding it would just memorize row identity, so it is dropped."
            )
            return ColumnDecision(
                column=col.name, missing_action="drop_column",
                encoding_action="none", scaling_action="none", reasons=reasons,
            )

        # --- Missing value handling ---
        if col.missing_pct >= MISSING_DROP_THRESHOLD:
            missing_action = "drop_column"
            reasons.append(
                f"{col.missing_pct}% missing exceeds the {MISSING_DROP_THRESHOLD}% threshold — "
                f"imputing this much data would be unreliable, so the column is dropped."
            )
        elif col.missing_pct > 0:
            if col.dtype == "numeric":
                missing_action = "impute_median"
                reasons.append(
                    f"{col.missing_pct}% missing — median imputation preserves the row and "
                    f"is robust to outliers, unlike mean imputation."
                )
            else:
                missing_action = "impute_mode"
                reasons.append(
                    f"{col.missing_pct}% missing — imputing with the most frequent category "
                    f"keeps the row without inventing a numeric value."
                )
        else:
            missing_action = "none"

        # --- Encoding (categorical only) ---
        encoding_action = "none"
        if col.dtype == "categorical" and missing_action != "drop_column":
            if col.is_high_cardinality:
                encoding_action = "frequency_encode"
                reasons.append(
                    f"{col.n_unique} unique categories is high-cardinality — one-hot encoding "
                    f"would explode the feature space, so frequency encoding is used instead."
                )
            else:
                encoding_action = "one_hot"
                reasons.append(
                    f"{col.n_unique} unique categories is low enough for one-hot encoding "
                    f"without blowing up dimensionality."
                )

        # --- Scaling (numeric only) ---
        scaling_action = "none"
        if col.dtype == "numeric" and missing_action != "drop_column":
            scaling_action = "standard_scale"
            reasons.append(
                "Numeric column — standardized so distance- and gradient-based models "
                "aren't dominated by columns with larger raw scales."
            )

        return ColumnDecision(
            column=col.name, missing_action=missing_action,
            encoding_action=encoding_action, scaling_action=scaling_action, reasons=reasons,
        )

    def _build_step_trace(self, decisions: List[ColumnDecision]) -> List[str]:
        trace = []
        for d in decisions:
            for reason in d.reasons:
                trace.append(f"[{d.column}] {reason}")
        return trace

    def _build_summary(self, decisions: List[ColumnDecision]) -> List[str]:
        dropped = [d.column for d in decisions if d.missing_action == "drop_column"]
        imputed = [d.column for d in decisions if d.missing_action.startswith("impute")]
        encoded = [d.column for d in decisions if d.encoding_action != "none"]
        scaled = [d.column for d in decisions if d.scaling_action != "none"]

        summary = []
        if dropped:
            summary.append(f"{len(dropped)} column(s) dropped (missingness or identifier): {dropped}")
        if imputed:
            summary.append(f"{len(imputed)} column(s) imputed: {imputed}")
        if encoded:
            summary.append(f"{len(encoded)} categorical column(s) encoded: {encoded}")
        if scaled:
            summary.append(f"{len(scaled)} numeric column(s) scheduled for scaling")
        if not summary:
            summary.append("No preprocessing needed — dataset is already clean.")
        return summary
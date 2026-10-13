"""
Target Preparation
Turns whatever column the user picks as the target into something a
classifier can learn from, and records every decision (with its reason)
as a plain-language note:
  - rows with no target value are removed
  - ID-like targets are rejected
  - continuous numeric targets are split into equal-sized groups
  - rare classes are merged into one "Other" class (or dropped if too few)
  - features that duplicate the target (leakage) are excluded
"""

from typing import List, Tuple

import numpy as np
import pandas as pd

MIN_CLASS_COUNT = 10
MAX_CLASSES = 15
N_BINS = 4
CONTINUOUS_UNIQUE_THRESHOLD = 20
ID_LIKE_RATIO = 0.5
LEAKAGE_PURITY = 0.99
LEAKAGE_MIN_LIFT = 0.05


class InvalidTargetError(Exception):
    """Raised when the chosen column cannot be turned into a usable target."""


def prepare_target(
    df: pd.DataFrame,
    target: str,
    n_bins: int = N_BINS,
    min_class_count: int = MIN_CLASS_COUNT,
    max_classes: int = MAX_CLASSES,
) -> Tuple[pd.DataFrame, List[str]]:
    if target not in df.columns:
        raise InvalidTargetError(f"'{target}' is not a column in this dataset.")

    out = df.copy()
    notes: List[str] = []

    n_missing = int(out[target].isna().sum())
    if n_missing:
        out = out.dropna(subset=[target])
        notes.append(
            f"{n_missing} row(s) had no value in '{target}' and were removed, "
            f"because a row without an answer cannot be used to learn or to test."
        )

    n_rows = len(out)
    if n_rows == 0:
        raise InvalidTargetError(f"'{target}' has no values at all.")

    series = out[target]
    n_unique = int(series.nunique())
    if n_unique < 2:
        raise InvalidTargetError(
            f"'{target}' has only {n_unique} distinct value, so there is nothing to predict."
        )

    is_numeric = pd.api.types.is_numeric_dtype(series)

    if not is_numeric and n_unique / n_rows > ID_LIKE_RATIO:
        raise InvalidTargetError(
            f"'{target}' has {n_unique} distinct values across {n_rows} rows, so almost "
            f"every row has its own value. It looks like an identifier or free text, "
            f"which cannot be predicted from other columns."
        )

    if is_numeric and n_unique > CONTINUOUS_UNIQUE_THRESHOLD:
        out[target], bin_notes = _bin_numeric(series, target, n_bins)
        notes.extend(bin_notes)
    else:
        out, class_notes = _clean_classes(out, target, min_class_count, max_classes)
        notes.extend(class_notes)

    out[target] = out[target].astype(str)

    out, leak_notes = _drop_leaking_features(out, target)
    notes.extend(leak_notes)

    order = sorted(out[target].unique())
    notes.append(
        "Class order used in the confusion matrices and metrics: "
        + ", ".join(f"{i} = {c}" for i, c in enumerate(order))
    )
    return out, notes


def _bin_numeric(series: pd.Series, target: str, n_bins: int) -> Tuple[pd.Series, List[str]]:
    notes: List[str] = []
    try:
        codes, edges = pd.qcut(series, q=n_bins, labels=False, retbins=True, duplicates="drop")
    except ValueError:
        codes, edges = None, None

    if codes is not None and len(edges) - 1 >= 2:
        k = len(edges) - 1
        labels = {i: f"Q{i + 1}: {edges[i]:.4g} to {edges[i + 1]:.4g}" for i in range(k)}
        notes.append(
            f"'{target}' is numeric with {series.nunique()} distinct values, so it is a "
            f"continuous measurement rather than a set of categories. It was split into "
            f"{k} equal-sized groups so it can be predicted as a classification problem. "
            f"This is a conversion, not regression: the model predicts which group a row "
            f"falls into."
        )
        return codes.map(labels), notes

    top = series.mode().iloc[0]
    share = float((series == top).mean())
    top_text = f"{float(top):.4g}"
    notes.append(
        f"'{target}' is numeric, but {share:.0%} of rows share the single value {top_text}, "
        f"so equal-sized groups are impossible. It was split into 'equals {top_text}' "
        f"versus 'not {top_text}'."
    )
    result = pd.Series(
        np.where(series == top, f"equals {top_text}", f"not {top_text}"),
        index=series.index,
    )
    return result, notes


def _clean_classes(
    df: pd.DataFrame, target: str, min_class_count: int, max_classes: int
) -> Tuple[pd.DataFrame, List[str]]:
    notes: List[str] = []
    series = df[target]
    counts = series.value_counts()

    keep = list(counts[counts >= min_class_count].index)
    if len(keep) > max_classes:
        keep = list(counts.index[:max_classes])
        notes.append(
            f"'{target}' has {len(counts)} distinct values. Only the {max_classes} most "
            f"frequent are kept as separate classes, to keep the comparison readable."
        )
    if len(keep) < 2:
        raise InvalidTargetError(
            f"Fewer than 2 values of '{target}' appear at least {min_class_count} times, "
            f"so there are not enough examples per class to learn from."
        )

    rare_mask = ~series.isin(keep)
    n_rare_rows = int(rare_mask.sum())
    if n_rare_rows:
        n_rare_values = int(series[rare_mask].nunique())
        if n_rare_rows >= min_class_count:
            df = df.copy()
            df[target] = series.astype(object).where(~rare_mask, "Other (merged)")
            notes.append(
                f"{n_rare_values} rare value(s) of '{target}' ({n_rare_rows} rows in total) "
                f"were merged into one 'Other (merged)' class, because every class needs "
                f"enough rows to appear in each cross-validation fold."
            )
        else:
            df = df[~rare_mask].copy()
            notes.append(
                f"{n_rare_values} rare value(s) of '{target}' ({n_rare_rows} rows in total) "
                f"were removed, because there are too few of them to learn from or to test on."
            )

    if df[target].nunique() < 2:
        raise InvalidTargetError(f"After cleaning, '{target}' has fewer than 2 usable classes.")
    return df, notes


def _drop_leaking_features(df: pd.DataFrame, target: str) -> Tuple[pd.DataFrame, List[str]]:
    notes: List[str] = []
    y = df[target]
    n_rows = len(df)
    majority_share = float(y.value_counts(normalize=True).iloc[0])
    limit = max(2, min(50, int(0.05 * n_rows)))

    leaking = []
    for col in df.columns:
        if col == target:
            continue
        n_values = int(df[col].nunique(dropna=True))
        if n_values < 2 or n_values > limit:
            continue
        table = pd.crosstab(df[col].astype(str), y)
        purity = float(table.max(axis=1).sum() / table.values.sum())
        if purity >= LEAKAGE_PURITY and (purity - majority_share) >= LEAKAGE_MIN_LIFT:
            leaking.append((col, purity))

    if leaking:
        df = df.drop(columns=[c for c, _ in leaking])
        for col, purity in leaking:
            notes.append(
                f"Feature '{col}' was excluded: its values alone identify the target class in "
                f"{purity:.1%} of rows (the most common class alone would be right "
                f"{majority_share:.1%} of the time). It duplicates the target, and keeping it "
                f"would make every score meaningless (target leakage)."
            )
    return df, notes
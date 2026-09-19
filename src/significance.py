"""
Statistical significance testing — checks whether the winning model's edge
over the runner-up is real or could plausibly be noise, using per-fold
cross-validation scores and a paired t-test.
"""
from typing import List, Tuple
from scipy import stats
import numpy as np


def paired_significance_test(fold_scores_a: List[float], fold_scores_b: List[float], alpha: float = 0.05) -> Tuple[float, float, bool]:
    """Returns (t_statistic, p_value, is_significant) for two models' per-fold scores."""
    t_stat, p_value = stats.ttest_rel(fold_scores_a, fold_scores_b)
    return float(t_stat), float(p_value), bool(p_value < alpha)
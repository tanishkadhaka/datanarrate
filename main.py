"""
Quick manual test — run this to confirm Module 1 works end to end.
Loads a real classification dataset and prints its profile.
"""
import json
import pandas as pd
from sklearn.datasets import load_breast_cancer
from src.profiler import DataProfiler


def load_sample_dataset() -> pd.DataFrame:
    data = load_breast_cancer(as_frame=True)
    return data.frame  # includes the 'target' column already


if __name__ == "__main__":
    df = load_sample_dataset()
    profiler = DataProfiler(df, target_column="target")
    result = profiler.profile()
    print(json.dumps(result.to_dict(), indent=2, default=str))
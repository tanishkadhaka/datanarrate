"""
Dataset loader. Default: Telco Customer Churn (7,043 rows, 21 raw columns,
11 missing values, real business classification problem). Alternates for
the multi-dataset evaluation phase are included below, commented out.
"""
import pandas as pd
from sklearn.datasets import fetch_openml

TELCO_URL = (
    "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/"
    "master/data/Telco-Customer-Churn.csv"
)


def load_telco_churn() -> pd.DataFrame:
    df = pd.read_csv(TELCO_URL)
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    return df


# --- Alternates for your evaluation phase (5-6 dataset comparison) ---

def load_satimage() -> pd.DataFrame:
    """6,435 rows, 36 numeric columns, zero missing, 6-class classification."""
    data = fetch_openml(name="satimage", version=1, as_frame=True)
    df = data.frame
    df.rename(columns={df.columns[-1]: "target"}, inplace=True)
    return df


def load_mushroom() -> pd.DataFrame:
    """8,124 rows, 22 categorical columns, ~30% missing in one column only."""
    data = fetch_openml(name="mushroom", version=1, as_frame=True)
    df = data.frame
    df.rename(columns={df.columns[-1]: "target"}, inplace=True)
    return df
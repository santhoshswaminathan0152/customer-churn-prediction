"""Download and clean the IBM Telco Customer Churn dataset."""
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data" / "telco_churn.csv"
URL = "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv"
TARGET = "Churn"


def download(path: Path = DATA_PATH) -> Path:
    if not path.exists():
        path.parent.mkdir(exist_ok=True)
        resp = requests.get(URL, timeout=60)
        resp.raise_for_status()
        path.write_bytes(resp.content)
    return path


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Fix types, fill TotalCharges blanks and encode the target as 0/1."""
    # Deduplicate on the customer ID only: different customers can legitimately share identical attributes.
    if "customerID" in df:
        df = df.drop_duplicates(subset="customerID")
    df = df.drop(columns=["customerID"], errors="ignore").copy()
    # TotalCharges is text with blanks for brand-new customers (tenure 0): they have not been billed yet.
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"].astype(str).str.strip(), errors="coerce")
    df.loc[df["TotalCharges"].isna() & (df["tenure"] == 0), "TotalCharges"] = 0.0
    df["TotalCharges"] = df["TotalCharges"].fillna(df["TotalCharges"].median())
    df["SeniorCitizen"] = df["SeniorCitizen"].map({1: "Yes", 0: "No"}).fillna(df["SeniorCitizen"].astype(str))
    if TARGET in df:
        df[TARGET] = (df[TARGET] == "Yes").astype(int)
    return df.reset_index(drop=True)


def load() -> pd.DataFrame:
    return clean(pd.read_csv(download()))

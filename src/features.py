"""Feature engineering + preprocessing, packaged so the API can score raw customer records."""
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

SERVICES = ["PhoneService", "MultipleLines", "OnlineSecurity", "OnlineBackup", "DeviceProtection",
            "TechSupport", "StreamingTV", "StreamingMovies"]
CATEGORICAL = ["gender", "SeniorCitizen", "Partner", "Dependents", "PhoneService", "MultipleLines",
               "InternetService", "OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport",
               "StreamingTV", "StreamingMovies", "Contract", "PaperlessBilling", "PaymentMethod", "tenure_bucket"]
NUMERIC = ["tenure", "MonthlyCharges", "TotalCharges", "num_services", "avg_monthly_spend", "charge_increase",
           "has_family", "auto_payment"]


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["tenure_bucket"] = pd.cut(df["tenure"], bins=[-1, 6, 12, 24, 48, np.inf],
                                 labels=["0-6m", "7-12m", "13-24m", "25-48m", "49m+"]).astype(str)
    df["num_services"] = (df[SERVICES] == "Yes").sum(axis=1) + (df["InternetService"] != "No")
    df["avg_monthly_spend"] = np.where(df["tenure"] > 0, df["TotalCharges"] / df["tenure"].clip(lower=1),
                                       df["MonthlyCharges"])
    # Current bill vs historical average: a recent price increase is a classic churn trigger.
    df["charge_increase"] = df["MonthlyCharges"] - df["avg_monthly_spend"]
    df["has_family"] = ((df["Partner"] == "Yes") | (df["Dependents"] == "Yes")).astype(int)
    df["auto_payment"] = df["PaymentMethod"].str.contains("automatic").astype(int)
    return df


def make_preprocessor(scale: bool = True):
    feature_step = FunctionTransformer(add_features)
    columns = ColumnTransformer([
        ("num", StandardScaler() if scale else "passthrough", NUMERIC),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL),
    ], verbose_feature_names_out=False)
    return feature_step, columns

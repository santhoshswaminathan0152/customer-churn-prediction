import pandas as pd

from src.data import clean
from src.features import add_features


def raw_rows():
    base = {"customerID": "A", "gender": "Female", "SeniorCitizen": 0, "Partner": "Yes", "Dependents": "No",
            "tenure": 0, "PhoneService": "Yes", "MultipleLines": "No", "InternetService": "DSL",
            "OnlineSecurity": "Yes", "OnlineBackup": "No", "DeviceProtection": "No", "TechSupport": "No",
            "StreamingTV": "No", "StreamingMovies": "No", "Contract": "Month-to-month", "PaperlessBilling": "Yes",
            "PaymentMethod": "Electronic check", "MonthlyCharges": 50.0, "TotalCharges": " ", "Churn": "Yes"}
    return pd.DataFrame([base, {**base, "customerID": "B", "tenure": 10, "TotalCharges": "400", "Churn": "No"},
                         {**base, "customerID": "B", "tenure": 10, "TotalCharges": "400", "Churn": "No"}])


def test_clean_fixes_blank_total_charges_and_target():
    df = clean(raw_rows())
    assert len(df) == 2  # duplicate customer ID removed
    assert df["TotalCharges"].tolist() == [0.0, 400.0]
    assert df["Churn"].tolist() == [1, 0]
    assert "customerID" not in df


def test_engineered_features():
    df = add_features(clean(raw_rows()))
    assert df["tenure_bucket"].tolist() == ["0-6m", "7-12m"]
    assert df["avg_monthly_spend"].tolist() == [50.0, 40.0]
    assert df["charge_increase"].tolist() == [0.0, 10.0]
    assert df["num_services"].tolist() == [3, 3]  # phone + online security + internet

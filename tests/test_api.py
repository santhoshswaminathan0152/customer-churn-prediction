import pytest

from src.data import ROOT

if not (ROOT / "models" / "churn_model.joblib").exists():
    pytest.skip("train the model first: python -m src.train", allow_module_level=True)

from fastapi.testclient import TestClient

from api import app

client = TestClient(app)
CUSTOMER = {"gender": "Female", "SeniorCitizen": 0, "Partner": "No", "Dependents": "No", "tenure": 2,
            "PhoneService": "Yes", "MultipleLines": "No", "InternetService": "Fiber optic", "OnlineSecurity": "No",
            "OnlineBackup": "No", "DeviceProtection": "No", "TechSupport": "No", "StreamingTV": "Yes",
            "StreamingMovies": "Yes", "Contract": "Month-to-month", "PaperlessBilling": "Yes",
            "PaymentMethod": "Electronic check", "MonthlyCharges": 99.5, "TotalCharges": 199.0}


def test_health():
    assert client.get("/health").json()["status"] == "ok"


def test_high_risk_customer():
    body = client.post("/predict", json=CUSTOMER).json()
    assert body["will_churn"] and body["risk_level"] == "high"


def test_loyal_customer_is_low_risk():
    loyal = {**CUSTOMER, "tenure": 70, "Contract": "Two year", "InternetService": "DSL", "OnlineSecurity": "Yes",
             "TechSupport": "Yes", "PaymentMethod": "Credit card (automatic)", "MonthlyCharges": 60.0,
             "TotalCharges": 4200.0}
    body = client.post("/predict", json=loyal).json()
    assert not body["will_churn"] and body["risk_level"] == "low"


def test_invalid_input_rejected():
    assert client.post("/predict", json={**CUSTOMER, "Contract": "Weekly"}).status_code == 422

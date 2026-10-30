"""FastAPI service that scores a customer's churn risk.

Run: uvicorn api:app --reload   then open http://127.0.0.1:8000/docs
"""
from typing import Literal

import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, Field

from src.data import ROOT, clean

YesNo = Literal["Yes", "No"]
bundle = joblib.load(ROOT / "models" / "churn_model.joblib")
app = FastAPI(title="Customer Churn Prediction API", version="1.0")


class Customer(BaseModel):
    gender: Literal["Male", "Female"]
    SeniorCitizen: Literal[0, 1]
    Partner: YesNo
    Dependents: YesNo
    tenure: int = Field(ge=0, le=100, description="Months with the company")
    PhoneService: YesNo
    MultipleLines: Literal["Yes", "No", "No phone service"]
    InternetService: Literal["DSL", "Fiber optic", "No"]
    OnlineSecurity: Literal["Yes", "No", "No internet service"]
    OnlineBackup: Literal["Yes", "No", "No internet service"]
    DeviceProtection: Literal["Yes", "No", "No internet service"]
    TechSupport: Literal["Yes", "No", "No internet service"]
    StreamingTV: Literal["Yes", "No", "No internet service"]
    StreamingMovies: Literal["Yes", "No", "No internet service"]
    Contract: Literal["Month-to-month", "One year", "Two year"]
    PaperlessBilling: YesNo
    PaymentMethod: Literal["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"]
    MonthlyCharges: float = Field(ge=0)
    TotalCharges: float = Field(ge=0)


class Prediction(BaseModel):
    churn_probability: float
    will_churn: bool
    risk_level: Literal["low", "medium", "high"]
    threshold: float


@app.get("/health")
def health():
    return {"status": "ok", "model": bundle["model_name"]}


@app.post("/predict", response_model=Prediction)
def predict(customer: Customer):
    row = clean(pd.DataFrame([customer.model_dump()]))
    p = float(bundle["pipeline"].predict_proba(row)[0, 1])
    t = bundle["threshold"]
    risk = "high" if p >= 0.7 else "medium" if p >= t else "low"
    return Prediction(churn_probability=round(p, 4), will_churn=p >= t, risk_level=risk, threshold=t)

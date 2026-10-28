from pathlib import Path
from typing import Literal

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "artifacts" / "churn_pipeline.pkl"

# The notebook evaluated the classifier using a 0.35 probability threshold.
# Keep this threshold consistent with the threshold selected for evaluation.
CHURN_THRESHOLD = 0.35

app = FastAPI(
    title="Telco Customer Churn API",
    description="Serves the fitted Logistic Regression pipeline trained in svmchurn.ipynb.",
    version="2.0.0",
)

# Load the fitted pipeline once when the API starts. No synthetic training.
try:
    pipeline = joblib.load(MODEL_PATH)
    expected_features = list(
        pipeline.named_steps["preprocessor"].feature_names_in_
    )
except FileNotFoundError as exc:
    raise RuntimeError(
        f"Model artifact not found at {MODEL_PATH}. "
        "Run the notebook training/save cell first and place "
        "churn_pipeline.pkl inside the artifacts directory."
    ) from exc
except Exception as exc:
    raise RuntimeError(
        f"Could not load the fitted pipeline from {MODEL_PATH}: {exc}"
    ) from exc


class CustomerFeatures(BaseModel):
    """Raw Telco customer features; send original, unencoded values."""

    gender: Literal["Female", "Male"]
    SeniorCitizen: int = Field(ge=0, le=1)
    Partner: Literal["Yes", "No"]
    Dependents: Literal["Yes", "No"]
    tenure: int = Field(ge=0, le=72)
    PhoneService: Literal["Yes", "No"]
    MultipleLines: Literal["Yes", "No", "No phone service"]
    InternetService: Literal["DSL", "Fiber optic", "No"]
    OnlineSecurity: Literal["Yes", "No", "No internet service"]
    OnlineBackup: Literal["Yes", "No", "No internet service"]
    DeviceProtection: Literal["Yes", "No", "No internet service"]
    TechSupport: Literal["Yes", "No", "No internet service"]
    StreamingTV: Literal["Yes", "No", "No internet service"]
    StreamingMovies: Literal["Yes", "No", "No internet service"]
    Contract: Literal["Month-to-month", "One year", "Two year"]
    PaperlessBilling: Literal["Yes", "No"]
    PaymentMethod: Literal[
        "Electronic check",
        "Mailed check",
        "Bank transfer (automatic)",
        "Credit card (automatic)",
    ]
    MonthlyCharges: float = Field(ge=0)
    TotalCharges: float = Field(ge=0)


def preprocess_raw_customer(raw_df: pd.DataFrame) -> pd.DataFrame:
    """
    Reproduce the notebook's manual encoding before passing the row to
    the saved pipeline. The saved pipeline then performs its fitted
    preprocessing/scaling and Logistic Regression prediction.
    """
    data = raw_df.copy()

    if "customerID" in data.columns:
        data = data.drop(columns=["customerID"])

    if "TotalCharges" in data.columns:
        data["TotalCharges"] = pd.to_numeric(
            data["TotalCharges"], errors="coerce"
        )

    binary_cols = [
        "gender",
        "Partner",
        "Dependents",
        "PhoneService",
        "PaperlessBilling",
    ]
    for col in binary_cols:
        if col in data.columns:
            data[col] = data[col].replace(
                {"Yes": 1, "No": 0, "Male": 1, "Female": 0}
            )

    multi_cat_cols = [
        "MultipleLines",
        "InternetService",
        "OnlineSecurity",
        "OnlineBackup",
        "DeviceProtection",
        "TechSupport",
        "StreamingTV",
        "StreamingMovies",
        "Contract",
        "PaymentMethod",
    ]
    existing_cols = [col for col in multi_cat_cols if col in data.columns]
    if existing_cols:
        data = pd.get_dummies(
            data, columns=existing_cols, drop_first=True, dtype=int
        )

    bool_cols = data.select_dtypes(include="bool").columns
    if len(bool_cols):
        data[bool_cols] = data[bool_cols].astype(int)

    # Match the feature consolidation used in the notebook.
    no_internet_cols = [
        col for col in data.columns
        if col.endswith("_No internet service")
    ]
    if no_internet_cols:
        data["No_internet_service"] = (
            data[no_internet_cols].any(axis=1).astype(int)
        )
        data = data.drop(columns=no_internet_cols)

    if "MultipleLines_No phone service" in data.columns:
        data["No_phone_service"] = data[
            "MultipleLines_No phone service"
        ].astype(int)
        data = data.drop(columns=["MultipleLines_No phone service"])

    # Align exactly to the encoded feature columns seen during notebook fit.
    # Missing dummy columns become 0; unexpected columns are discarded.
    data = data.replace([np.inf, -np.inf], np.nan)
    data = data.reindex(columns=expected_features, fill_value=0)

    return data


@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "model_loaded": True,
        "model_file": MODEL_PATH.name,
        "classifier": type(pipeline.named_steps["model"]).__name__,
        "churn_threshold": CHURN_THRESHOLD,
    }


@app.get("/model-info")
def model_info():
    return {
        "model_file": MODEL_PATH.name,
        "classifier": type(pipeline.named_steps["model"]).__name__,
        "n_features": len(expected_features),
        "churn_threshold": CHURN_THRESHOLD,
    }


@app.post("/predict")
def predict_churn(customer: CustomerFeatures):
    try:
        raw_input = pd.DataFrame([customer.model_dump()])
        processed_input = preprocess_raw_customer(raw_input)

        # Use probabilities with the same 0.35 threshold evaluated in notebook.
        probability = float(pipeline.predict_proba(processed_input)[0, 1])
        prediction = int(probability >= CHURN_THRESHOLD)

        return {
            "churn_prediction": prediction,
            "churn_risk_label": (
                "High Churn Risk" if prediction == 1 else "Low Risk / Retain"
            ),
            "churn_probability": round(probability, 4),
            "decision_threshold": CHURN_THRESHOLD,
            "classifier": type(pipeline.named_steps["model"]).__name__,
        }
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {exc}",
        ) from exc

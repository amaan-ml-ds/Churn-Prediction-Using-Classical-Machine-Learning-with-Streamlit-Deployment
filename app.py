import os

import requests
import streamlit as st

API_URL = os.getenv("CHURN_API_URL", "http://127.0.0.1:8000")

st.set_page_config(
    page_title="Telco Churn Predictor",
    page_icon="📡",
    layout="wide",
)

st.title("📡 Telco Customer Churn Predictor")
st.caption("Streamlit frontend → FastAPI backend → saved notebook pipeline")

# Backend status and saved model information
try:
    health_response = requests.get(f"{API_URL}/health", timeout=3)
    health_response.raise_for_status()
    health = health_response.json()
    st.sidebar.success("FastAPI backend connected")
    st.sidebar.write(f"Classifier: **{health.get('classifier', 'Unknown')}**")
    st.sidebar.write(
        f"Churn threshold: **{health.get('churn_threshold', 'Unknown')}**"
    )
except requests.RequestException:
    st.sidebar.error("FastAPI backend is offline")
    st.sidebar.caption("Start the backend with: uvicorn main:app --reload")

st.markdown(
    "Enter a customer's details to estimate churn probability. "
    "The API loads the fitted `churn_pipeline.pkl` saved by the notebook; "
    "it does not train on synthetic data."
)

with st.form("customer_prediction_form"):
    st.subheader("Customer profile")

    col1, col2, col3 = st.columns(3)

    with col1:
        gender = st.selectbox("Gender", ["Female", "Male"])
        senior = st.selectbox("Senior citizen", [0, 1])
        partner = st.selectbox("Partner", ["Yes", "No"])
        dependents = st.selectbox("Dependents", ["No", "Yes"])
        tenure = st.slider("Tenure (months)", 0, 72, 12)
        phone_service = st.selectbox("Phone service", ["Yes", "No"])
        multiple_lines = st.selectbox(
            "Multiple lines", ["No", "Yes", "No phone service"]
        )

    with col2:
        internet = st.selectbox(
            "Internet service", ["DSL", "Fiber optic", "No"]
        )
        security = st.selectbox(
            "Online security", ["No", "Yes", "No internet service"]
        )
        backup = st.selectbox(
            "Online backup", ["No", "Yes", "No internet service"]
        )
        protection = st.selectbox(
            "Device protection", ["No", "Yes", "No internet service"]
        )
        support = st.selectbox(
            "Tech support", ["No", "Yes", "No internet service"]
        )
        tv = st.selectbox(
            "Streaming TV", ["No", "Yes", "No internet service"]
        )
        movies = st.selectbox(
            "Streaming movies", ["No", "Yes", "No internet service"]
        )

    with col3:
        contract = st.selectbox(
            "Contract", ["Month-to-month", "One year", "Two year"]
        )
        paperless = st.selectbox("Paperless billing", ["Yes", "No"])
        payment = st.selectbox(
            "Payment method",
            [
                "Electronic check",
                "Mailed check",
                "Bank transfer (automatic)",
                "Credit card (automatic)",
            ],
        )
        monthly = st.number_input(
            "Monthly charges", min_value=0.0, value=70.35, step=1.0
        )
        total = st.number_input(
            "Total charges", min_value=0.0, value=844.20, step=10.0
        )

    submitted = st.form_submit_button(
        "Predict churn risk", type="primary", use_container_width=True
    )

if submitted:
    payload = {
        "gender": gender,
        "SeniorCitizen": senior,
        "Partner": partner,
        "Dependents": dependents,
        "tenure": tenure,
        "PhoneService": phone_service,
        "MultipleLines": multiple_lines,
        "InternetService": internet,
        "OnlineSecurity": security,
        "OnlineBackup": backup,
        "DeviceProtection": protection,
        "TechSupport": support,
        "StreamingTV": tv,
        "StreamingMovies": movies,
        "Contract": contract,
        "PaperlessBilling": paperless,
        "PaymentMethod": payment,
        "MonthlyCharges": monthly,
        "TotalCharges": total,
    }

    try:
        with st.spinner("Requesting prediction..."):
            response = requests.post(
                f"{API_URL}/predict", json=payload, timeout=15
            )
        response.raise_for_status()
        result = response.json()

        st.subheader("Prediction result")
        probability = result["churn_probability"] * 100

        metric_col, status_col = st.columns(2)
        metric_col.metric("Estimated churn probability", f"{probability:.1f}%")
        metric_col.caption(
            f"Decision threshold: {result['decision_threshold']:.0%}"
        )

        if result["churn_prediction"] == 1:
            status_col.error(f"⚠️ {result['churn_risk_label']}")
        else:
            status_col.success(f"✅ {result['churn_risk_label']}")

        st.caption(
            f"Classifier: {result.get('classifier', 'Unknown')}"
        )

    except requests.RequestException as exc:
        st.error(f"Could not get a prediction from the API: {exc}")
        if getattr(exc, "response", None) is not None:
            st.code(exc.response.text)

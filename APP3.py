import streamlit as st
import joblib
import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset
import plotly.graph_objects as go
import os
from datetime import datetime

# Load artifacts
@st.cache_resource
def load_artifacts():
    model = joblib.load("loan_default_model.pkl")
    preprocessor = joblib.load("preprocessor.pkl")
    reference_data = pd.read_csv("reference_data.csv") if os.path.exists("reference_data.csv") else None
    return {"model": model, "preprocessor": preprocessor, "reference_data": reference_data}

artifacts = load_artifacts()

# Setup SHAP explainer (Tree-based model assumed)
@st.cache_resource
def get_explainer():
    try:
        background = artifacts["preprocessor"].transform(
            artifacts["reference_data"].drop(columns=["Default"])
        ) if artifacts["reference_data"] is not None else None
        explainer = shap.Explainer(artifacts["model"], background)
        return explainer
    except Exception as e:
        st.warning(f"SHAP explainer setup failed: {e}")
        return None

explainer = get_explainer()

# Logging function (ethical: anonymized only predictions)
def log_prediction(prediction: int, probability: float):
    if not os.path.exists("logs"):
        os.makedirs("logs")
    with open("logs/predictions.log", "a") as f:
        f.write(f"{datetime.now()}, Prediction: {prediction}, Prob: {probability:.4f}\n")

# Streamlit App
st.set_page_config(page_title="Loan Default Prediction App", layout="wide")
st.title("💳 Loan Default Prediction App")

menu = ["Home", "Predict (Single)", "Predict (Batch)", "Drift Report", "Logs", "Model Explainability"]
choice = st.sidebar.radio("Navigation", menu)

# Home
if choice == "Home":
    st.subheader("Welcome")
    st.write("This app predicts loan default risk using an ML model with preprocessing pipeline.")
    st.markdown("### Features\n- Single & batch predictions\n- Data drift monitoring\n- Prediction logs\n- SHAP explainability (new)")

# Single Prediction
elif choice == "Predict (Single)":
    st.subheader("Single Prediction")
    with st.form("input_form"):
        features = {}
        for col in ["Age","Income","LoanAmount","CreditScore","MonthsEmployed","NumCreditLines","InterestRate","LoanTerm","DTIRatio","Education","EmploymentType","MaritalStatus","HasMortgage","HasDependents","LoanPurpose","HasCoSigner"]:
            features[col] = st.text_input(f"Enter {col}")
        submit = st.form_submit_button("Predict")

    if submit:
        input_df = pd.DataFrame([features])
        input_df = input_df.replace("", np.nan).astype(str)
        try:
            X = artifacts["preprocessor"].transform(input_df)
            pred = artifacts["model"].predict(X)[0]
            prob = artifacts["model"].predict_proba(X)[0][1]
            st.success(f"Prediction: {'Default' if pred==1 else 'No Default'} (Prob: {prob:.2f})")
            log_prediction(pred, prob)

            # Gauge visualization
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=prob*100,
                title={'text': "Default Probability (%)"},
                gauge={'axis': {'range': [0,100]}, 'bar': {'color': "red" if prob>0.5 else "green"}}
            ))
            st.plotly_chart(fig, use_container_width=True)

            # Save latest input for SHAP
            input_df.to_csv("latest_input.csv", index=False)

        except Exception as e:
            st.error(f"Error: {e}")

# Batch Prediction
elif choice == "Predict (Batch)":
    st.subheader("Batch Prediction")
    uploaded = st.file_uploader("Upload CSV", type="csv")
    if uploaded:
        data = pd.read_csv(uploaded)
        try:
            X = artifacts["preprocessor"].transform(data)
            preds = artifacts["model"].predict(X)
            data["Prediction"] = preds
            st.dataframe(data.head())
            st.download_button("Download Predictions", data.to_csv(index=False), "predictions.csv")
        except Exception as e:
            st.error(f"Error: {e}")

# Drift Report
elif choice == "Drift Report":
    st.subheader("Data Drift Report")
    if artifacts["reference_data"] is not None:
        uploaded = st.file_uploader("Upload new data for drift analysis", type="csv")
        if uploaded:
            new_data = pd.read_csv(uploaded)
            report = Report(metrics=[DataDriftPreset()])
            report.run(reference_data=artifacts["reference_data"].drop(columns=["Default"]), current_data=new_data)
            report.save_html("drift_report.html")
            with open("drift_report.html","r",encoding="utf-8") as f:
                st.components.v1.html(f.read(), height=800, scrolling=True)
    else:
        st.info("Reference data not available for drift analysis.")

# Logs
elif choice == "Logs":
    st.subheader("Prediction Logs")
    if os.path.exists("logs/predictions.log"):
        with open("logs/predictions.log") as f:
            st.text(f.read())
    else:
        st.info("No logs yet.")

# Model Explainability
elif choice == "Model Explainability":
    st.subheader("Model Explainability with SHAP")
    if explainer is None:
        st.warning("SHAP explainer not available.")
    else:
        if os.path.exists("latest_input.csv"):
            latest_input = pd.read_csv("latest_input.csv")
            X = artifacts["preprocessor"].transform(latest_input)
            shap_values = explainer(X)

            st.write("### Feature Importance (Latest Prediction)")
            fig, ax = plt.subplots()
            shap.plots.bar(shap_values[0], show=False)
            st.pyplot(fig)

            st.write("### Detailed Impact (Waterfall Plot)")
            fig, ax = plt.subplots()
            shap.plots.waterfall(shap_values[0], show=False)
            st.pyplot(fig)
        else:
            st.info("Run a prediction first to generate SHAP explanations.")

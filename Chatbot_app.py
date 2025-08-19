# app.py
import streamlit as st
import pandas as pd
import joblib
import numpy as np

# ---------------------------
# Load model & preprocessor
# ---------------------------
@st.cache_resource
def load_model():
    model = joblib.load("loan_default_model.pkl")
    preprocessor = joblib.load("preprocessor.pkl")
    return model, preprocessor

model, preprocessor = load_model()
features = preprocessor.feature_names_in_

# ---------------------------
# Title
# ---------------------------
st.title("💰 Loan & Financial Advisory Chatbot")
st.write("Assess loan default risk with interactive explanations.")

# ---------------------------
# Input Mode
# ---------------------------
input_mode = st.sidebar.selectbox("Choose Input Mode", ["Single Applicant", "CSV Batch Upload"])

# =========================
# Single Applicant Input
# =========================
if input_mode == "Single Applicant":
    st.header("Enter Applicant Details")
    user_input = {}

    for f in features:
        if "Age" in f or "Income" in f or "LoanAmount" in f or "CreditScore" in f or \
           "MonthsEmployed" in f or "NumCreditLines" in f or "InterestRate" in f or \
           "LoanTerm" in f or "DTIRatio" in f:
            user_input[f] = st.number_input(f, value=0)
        else:
            user_input[f] = st.selectbox(f, ["Low", "Medium", "High"])

    if st.button("Predict Risk"):
        input_df = pd.DataFrame([user_input])
        X = preprocessor.transform(input_df)
        pred_proba = model.predict_proba(X)[:,1][0]
        st.subheader(f"Predicted Default Risk: {pred_proba*100:.2f}%")

        # ---------------------------
        # Feature impact explanation (alternative to SHAP)
        # ---------------------------
        st.subheader("Feature Impact Explanation")
        mean_vals = np.mean(preprocessor.transform(pd.DataFrame(
            np.random.randn(100, len(features)), columns=features)), axis=0)
        input_vals = X[0]
        impact_scores = input_vals - mean_vals
        impact_df = pd.DataFrame({"Feature": features, "ImpactScore": impact_scores})
        impact_df["ImpactAbs"] = impact_df["ImpactScore"].abs()
        impact_df = impact_df.sort_values(by="ImpactAbs", ascending=False)
        st.table(impact_df.head(5))

        # ---------------------------
        # Recommendations
        # ---------------------------
        st.subheader("Recommendations:")
        if user_input.get("DTIRatio",0) > 0.4:
            st.write("- Consider reducing your existing debt.")
        if user_input.get("CreditScore",0) < 650:
            st.write("- Work on improving your credit score.")
        if user_input.get("MonthsEmployed",0) < 12:
            st.write("- Stable employment history improves chances.")
        st.write("These suggestions are informational only, not legal or financial advice.")

# =========================
# CSV Batch Upload
# =========================
if input_mode == "CSV Batch Upload":
    st.header("Upload CSV for Batch Prediction")
    uploaded_file = st.file_uploader("Upload CSV", type=["csv"])
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        st.write("Preview of uploaded data:")
        st.dataframe(df.head())

        if st.button("Predict Batch Risk"):
            try:
                X_batch = preprocessor.transform(df)
                pred_probs = model.predict_proba(X_batch)[:,1]
                df["DefaultRisk"] = pred_probs
                st.subheader("Predictions:")
                st.dataframe(df)

                # Feature impact explanation (batch)
                mean_vals = np.mean(X_batch, axis=0)
                impact_scores = np.mean(np.abs(X_batch - mean_vals), axis=0)
                impact_df = pd.DataFrame({"Feature": features, "MeanImpact": impact_scores})
                impact_df = impact_df.sort_values(by="MeanImpact", ascending=False)
                st.subheader("Top Contributing Features (Batch Average):")
                st.dataframe(impact_df.head(5))

                # Download predictions
                csv = df.to_csv(index=False).encode('utf-8')
                st.download_button("Download Predictions CSV", csv, "batch_predictions.csv", "text/csv")
            except Exception as e:
                st.error(f"Error processing CSV: {e}")

# APP3.py
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import os
import traceback
from datetime import datetime
from sklearn.base import BaseEstimator, TransformerMixin

# Evidently for monitoring
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset

# ===== Configuration =====
MODEL_PATH = 'loan_default_model.pkl'
PREPROCESSOR_PATH = 'preprocessor.pkl'
REFERENCE_DATA_PATH = "X_train.csv"
DEVELOPER_NAME = "Jibraan Attar"
MODEL_VERSION = "2.0"
MODEL_TRAIN_DATE = "2025-07-29"
MAX_CSV_ROWS = 2000000
MAX_FILE_SIZE_MB = 25

# Features
NUMERICAL_FEATURES = [
    'Age', 'Income', 'LoanAmount', 'CreditScore', 'MonthsEmployed',
    'NumCreditLines', 'InterestRate', 'LoanTerm', 'DTIRatio'
]
CATEGORICAL_FEATURES = ['Education', 'EmploymentType', 'LoanPurpose']
BINARY_FEATURES = ['MaritalStatus', 'HasMortgage', 'HasDependents', 'HasCoSigner']
FEATURES_ORDER = NUMERICAL_FEATURES + CATEGORICAL_FEATURES + BINARY_FEATURES

# ===== Logging Setup =====
os.makedirs("logs", exist_ok=True)
import logging
logging.basicConfig(
    filename="logs/predictions.log",
    level=logging.INFO,
    format="%(asctime)s - %(message)s"
)

def log_prediction(input_data, prediction):
    """Log each prediction"""
    logging.info(f"Input: {input_data.to_dict(orient='records') if isinstance(input_data, pd.DataFrame) else input_data}, Prediction: {prediction}")

def generate_drift_report(reference_data, new_data, report_name="drift_report"):
    """Generate and save drift report with Evidently"""
    try:
        # ✅ Skip drift detection for too few rows
        if len(new_data) < 30:
            logging.info("Skipped drift report: not enough rows in current data")
            return None

        report = Report(metrics=[DataDriftPreset()])
        report.run(reference_data=reference_data, current_data=new_data)
        report_path = f"logs/{report_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
        report.save_html(report_path)
        logging.info(f"Drift report saved: {report_path}")
        return report_path
    except Exception as e:
        logging.error(f"Error generating drift report: {str(e)}")
        return None

# ===== Preprocessor Class =====
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, OrdinalEncoder

class Preprocessor(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.column_transformer = ColumnTransformer(
            transformers=[
                ('num', StandardScaler(), NUMERICAL_FEATURES),
                ('cat', OneHotEncoder(drop='first', sparse_output=False), CATEGORICAL_FEATURES),
                ('bin', OrdinalEncoder(), BINARY_FEATURES)
            ]
        )
    def fit(self, X, y=None):
        self.column_transformer.fit(X)
        return self
    def transform(self, X):
        return self.column_transformer.transform(X)

# ===== File Validation =====
if not os.path.exists(MODEL_PATH):
    st.error(f"Model file not found at {MODEL_PATH}")
    st.stop()
if not os.path.exists(PREPROCESSOR_PATH):
    st.error(f"Preprocessor file not found at {PREPROCESSOR_PATH}")
    st.stop()
if not os.path.exists(REFERENCE_DATA_PATH):
    st.error(f"Reference training data not found at {REFERENCE_DATA_PATH}")
    st.stop()

# ===== Helper Functions =====
@st.cache_resource
def load_artifacts():
    artifacts = {
        'model': joblib.load(MODEL_PATH),
        'preprocessor': joblib.load(PREPROCESSOR_PATH),
        'reference_data': pd.read_csv(REFERENCE_DATA_PATH)
    }
    return artifacts

def predict_default_probability(input_data, artifacts):
    try:
        for col in NUMERICAL_FEATURES:
            input_data[col] = pd.to_numeric(input_data[col], errors='coerce')
        processed = artifacts['preprocessor'].transform(input_data)
        probs = artifacts['model'].predict_proba(processed)
        return probs[:, 1]
    except Exception as e:
        st.error(f"Prediction error: {str(e)}")
        return None

def get_user_input():
    with st.form("loan_input"):
        st.header("Applicant Information")
        col1, col2, col3 = st.columns(3)
        age = col1.number_input("Age", 18, 100, 35)
        income = col2.number_input("Annual Income ($)", 1000, 500000)  # ✅ increased max
        loan_amount = col3.number_input("Loan Amount ($)", 1000, 200000)  # ✅ increased max
        col1, col2, col3 = st.columns(3)
        credit_score = col1.number_input("Credit Score", 300, 850, 700)
        months_employed = col2.number_input("Months Employed", 0, 480)  # ✅ up to 40 yrs
        num_credit_lines = col3.number_input("Number of Credit Lines", 0, 50)
        col1, col2, col3 = st.columns(3)
        interest_rate = col1.number_input("Interest Rate (%)", 0.0, 50.0, 7.5, step=0.1)
        loan_term = col2.number_input("Loan Term (months)", 1, 360)  # ✅ up to 30 yrs
        dti_ratio = col3.number_input("DTI Ratio", 0.0, 2.0, 0.35, step=0.01)
        col1, col2, col3 = st.columns(3)
        education = col1.selectbox("Education", ["High School", "Bachelor's", "Master's", "PhD"])
        employment_type = col2.selectbox("Employment Type", ["Full-time", "Part-time", "Self-employed", "Unemployed"])
        loan_purpose = col3.selectbox("Loan Purpose", ["Business", "Home", "Education", "Auto"])
        col1, col2, col3, col4 = st.columns(4)
        marital_status = col1.selectbox("Marital Status", ["Single", "Married"])
        has_mortgage = col2.selectbox("Has Mortgage", ["No", "Yes"])
        has_dependents = col3.selectbox("Has Dependents", ["No", "Yes"])
        has_cosigner = col4.selectbox("Has Co-signer", ["No", "Yes"])
        submitted = st.form_submit_button("Predict Default Risk")
        if submitted:
            return pd.DataFrame([{
                'Age': int(age), 'Income': float(income), 'LoanAmount': float(loan_amount),
                'CreditScore': int(credit_score), 'MonthsEmployed': int(months_employed),
                'NumCreditLines': int(num_credit_lines), 'InterestRate': float(interest_rate),
                'LoanTerm': int(loan_term), 'DTIRatio': float(dti_ratio), 'Education': education,
                'EmploymentType': employment_type, 'MaritalStatus': marital_status,
                'HasMortgage': has_mortgage, 'HasDependents': has_dependents,
                'LoanPurpose': loan_purpose, 'HasCoSigner': has_cosigner
            }])
    return None

def process_batch_data(uploaded_file, artifacts):
    df = pd.read_csv(uploaded_file, nrows=MAX_CSV_ROWS)
    probs = predict_default_probability(df[FEATURES_ORDER], artifacts)
    if probs is not None:
        df['Default_Probability'] = probs
        df['Risk_Classification'] = np.where(probs >= 0.5, "High Risk", "Low Risk")
        return df
    return None

# ===== Main App =====
def main():
    st.set_page_config(page_title="Loan Default Predictor", page_icon="💰", layout="wide")
    artifacts = load_artifacts()
    st.title("Loan Default Risk Assessment with Monitoring")

    tab1, tab2, tab3 = st.tabs(["Single Application", "Batch Processing", "Prediction Logs"])

    with tab1:
        input_df = get_user_input()
        if input_df is not None:
            probs = predict_default_probability(input_df, artifacts)
            if probs is not None:
                prob_default = float(probs[0])
                prediction = "High Risk" if prob_default >= 0.5 else "Low Risk"
                st.metric("Probability of Default", f"{prob_default:.2%}")
                st.metric("Risk Classification", prediction)

                # Log + Drift Report
                log_prediction(input_df, prediction)
                report_path = generate_drift_report(artifacts['reference_data'], input_df, "single_app")
                if report_path:
                    with open(report_path, "rb") as f:
                        st.download_button("Download Drift Report", f, file_name=os.path.basename(report_path))

    with tab2:
        uploaded_file = st.file_uploader("Upload CSV", type="csv")
        if uploaded_file is not None:
            results_df = process_batch_data(uploaded_file, artifacts)
            if results_df is not None:
                st.success(f"Processed {len(results_df)} applications!")
                log_prediction(results_df[FEATURES_ORDER], results_df['Risk_Classification'].tolist())
                report_path = generate_drift_report(artifacts['reference_data'], results_df[FEATURES_ORDER], "batch_app")
                if report_path:
                    with open(report_path, "rb") as f:
                        st.download_button("Download Batch Drift Report", f, file_name=os.path.basename(report_path))
                st.dataframe(results_df.head())

    with tab3:
        st.subheader("Prediction Logs")
        try:
            if os.path.exists("logs/predictions.log"):
                with open("logs/predictions.log", "r") as f:
                    log_content = f.read()
                st.text_area("Logs", log_content, height=300)
                with open("logs/predictions.log", "rb") as f:
                    st.download_button("Download Prediction Logs", f, file_name="predictions.log")
            else:
                st.info("No logs found yet.")
        except Exception as e:
            st.error(f"Error reading logs: {str(e)}")

if __name__ == "__main__":
    try:
        main()
    except Exception:
        st.error("Critical error occurred")
        st.code(traceback.format_exc())
        st.stop()

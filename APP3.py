# APP3.py
import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import numpy as np
import joblib
import os
import traceback
from datetime import datetime
from sklearn.base import BaseEstimator, TransformerMixin

# Plotly for interactive visuals
import plotly.graph_objects as go
import plotly.express as px

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
MAX_CSV_ROWS = 2_000_000
MAX_FILE_SIZE_MB = 25
MIN_DRIFT_ROWS = 30       # << skip drift if fewer rows than this
THRESHOLD = 0.50          # risk cut-off

# Feature definitions
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
    logging.info(
        f"Input: {input_data.to_dict(orient='records') if isinstance(input_data, pd.DataFrame) else input_data}, "
        f"Prediction: {prediction}"
    )

# ===== Utilities =====
def _strip_aux_columns(df: pd.DataFrame) -> pd.DataFrame:
    # Drop stray index/name columns like Unnamed: 0, LoanID, etc.
    cols = [c for c in df.columns if not c.startswith("Unnamed")]
    df = df[cols]
    # Keep only features we expect
    keep = [c for c in df.columns if c in FEATURES_ORDER]
    df = df[keep]
    # Reindex to bring columns in exact order (missing ones will be added as NaN)
    df = df.reindex(columns=FEATURES_ORDER)
    return df

def _drift_ready(reference_df: pd.DataFrame, current_df: pd.DataFrame):
    ref = _strip_aux_columns(reference_df.copy())
    cur = _strip_aux_columns(current_df.copy())
    # Basic type harmonization
    for col in NUMERICAL_FEATURES:
        if col in ref: ref[col] = pd.to_numeric(ref[col], errors='coerce')
        if col in cur: cur[col] = pd.to_numeric(cur[col], errors='coerce')
    for col in CATEGORICAL_FEATURES + BINARY_FEATURES:
        if col in ref: ref[col] = ref[col].astype(str)
        if col in cur: cur[col] = cur[col].astype(str)
    # Drop rows with all-NaN features to avoid Evidently errors
    ref = ref.dropna(how="all")
    cur = cur.dropna(how="all")
    return ref, cur

def generate_drift_report(reference_data, new_data, report_name="drift_report"):
    """Generate and save drift report with Evidently. Skips when too few rows."""
    try:
        # Prepare and align data
        reference_data, new_data = _drift_ready(reference_data, new_data)

        # Skip if too few rows
        if len(new_data) < MIN_DRIFT_ROWS:
            logging.info(f"Skipped drift report: only {len(new_data)} rows (< {MIN_DRIFT_ROWS}).")
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

# ===== Preprocessor Class (to make joblib load safe) =====
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
    # Clean the reference once
    artifacts['reference_data'] = _strip_aux_columns(artifacts['reference_data'])
    return artifacts

def predict_default_probability(input_data, artifacts):
    try:
        for col in NUMERICAL_FEATURES:
            if col in input_data:
                input_data[col] = pd.to_numeric(input_data[col], errors='coerce')
        processed = artifacts['preprocessor'].transform(input_data)
        probs = artifacts['model'].predict_proba(processed)
        return probs[:, 1]
    except Exception as e:
        st.error(f"Prediction error: {str(e)}")
        with st.expander("Traceback"):
            st.code(traceback.format_exc())
        return None

# ===== UI Builders (Plotly) =====
def gauge_probability(prob):
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=float(prob) * 100,
        number={'suffix': "%"},
        title={'text': "Default Probability"},
        gauge={
            "axis": {"range": [0, 100]},
            "bar": {"thickness": 0.35},
            "steps": [
                {"range": [0, THRESHOLD*100], "color": "#1f77b4"},
                {"range": [THRESHOLD*100, 100], "color": "#ff7f0e"},
            ],
            "threshold": {"line": {"width": 3}, "thickness": 0.85, "value": THRESHOLD*100}
        }
    ))
    fig.update_layout(height=280, margin=dict(l=10, r=10, t=40, b=10))
    return fig

def risk_pie(df):
    counts = df['Risk_Classification'].value_counts().reset_index()
    counts.columns = ['Risk', 'Count']
    fig = px.pie(counts, names='Risk', values='Count', hole=0.4, title='Risk Split')
    fig.update_layout(height=330, margin=dict(l=10, r=10, t=40, b=10))
    return fig

def prob_hist(df):
    fig = px.histogram(df, x='Default_Probability', nbins=30,
                       title='Predicted Probability Distribution')
    fig.update_layout(height=330, xaxis_title="Default Probability",
                      yaxis_title="Count", margin=dict(l=10, r=10, t=40, b=10))
    return fig

# ===== Forms and Batch =====
def get_user_input():
    with st.form("loan_input"):
        st.header("Applicant Information")
        col1, col2, col3 = st.columns(3)
        age = col1.number_input("Age", 18, 100, 35)
        income = col2.number_input("Annual Income ($)", 1000, 5_000_000, 60_000)  # increased max
        loan_amount = col3.number_input("Loan Amount ($)", 500, 2_000_000, 200_000)

        col1, col2, col3 = st.columns(3)
        credit_score = col1.number_input("Credit Score", 300, 850, 700)
        months_employed = col2.number_input("Months Employed", 0, 600, 36)
        num_credit_lines = col3.number_input("Number of Credit Lines", 0, 50, 3)

        col1, col2, col3 = st.columns(3)
        interest_rate = col1.number_input("Interest Rate (%)", 0.0, 50.0, 7.5, step=0.1)
        loan_term = col2.number_input("Loan Term (months)", 1, 480, 360)
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
    df = _strip_aux_columns(df)
    probs = predict_default_probability(df[FEATURES_ORDER], artifacts)
    if probs is not None:
        df['Default_Probability'] = probs
        df['Risk_Classification'] = np.where(probs >= THRESHOLD, "High Risk", "Low Risk")
        return df
    return None

# ===== Main App =====
def main():
    st.set_page_config(page_title="Loan Default Predictor", page_icon="💰", layout="wide")
    artifacts = load_artifacts()

    st.title("Loan Default Risk Assessment with Monitoring")
    st.caption(f"Developer: **{DEVELOPER_NAME}** | Model v{MODEL_VERSION} (trained {MODEL_TRAIN_DATE})")

    tab1, tab2, tab3 = st.tabs(["Single Application", "Batch Processing", "Prediction Logs"])

    # ---- Single Application
    with tab1:
        input_df = get_user_input()
        if input_df is not None:
            probs = predict_default_probability(input_df.copy(), artifacts)
            if probs is not None:
                p = float(probs[0])
                prediction = "High Risk" if p >= THRESHOLD else "Low Risk"

                # KPI row
                c1, c2, c3 = st.columns(3)
                c1.metric("Default Probability", f"{p:.2%}")
                c2.metric("Risk Classification", prediction)
                c3.metric("Threshold", f"{THRESHOLD:.0%}")

                # Gauge
                st.plotly_chart(gauge_probability(p), use_container_width=True)

                # Log
                log_prediction(input_df, prediction)

                # Drift (skips if < MIN_DRIFT_ROWS)
                report_path = generate_drift_report(artifacts['reference_data'], input_df, "single_app")
                if report_path:
                    st.success("Data drift report generated.")
                    with open(report_path, "rb") as f:
                        st.download_button("Download Drift Report (HTML)", f,
                                           file_name=os.path.basename(report_path), key="single_html")
                    # Inline preview
                    with open(report_path, "r", encoding="utf-8") as f:
                        html = f.read()
                    components.html(html, height=600, scrolling=True)
                else:
                    st.info(f"Drift check skipped (need at least {MIN_DRIFT_ROWS} rows).")

    # ---- Batch Processing
    with tab2:
        uploaded_file = st.file_uploader("Upload CSV", type="csv")
        if uploaded_file is not None:
            results_df = process_batch_data(uploaded_file, artifacts)
            if results_df is not None:
                st.success(f"Processed {len(results_df)} applications!")
                # KPIs
                high = int((results_df['Risk_Classification'] == "High Risk").sum())
                low = int((results_df['Risk_Classification'] == "Low Risk").sum())
                avg_p = float(results_df['Default_Probability'].mean())
                c1, c2, c3 = st.columns(3)
                c1.metric("High Risk", f"{high}")
                c2.metric("Low Risk", f"{low}")
                c3.metric("Avg Default Prob", f"{avg_p:.2%}")

                # Charts
                ch1, ch2 = st.columns(2)
                with ch1:
                    st.plotly_chart(risk_pie(results_df), use_container_width=True)
                with ch2:
                    st.plotly_chart(prob_hist(results_df), use_container_width=True)

                # Log all rows
                log_prediction(results_df[FEATURES_ORDER], results_df['Risk_Classification'].tolist())

                # Drift
                report_path = generate_drift_report(
                    artifacts['reference_data'],
                    results_df[FEATURES_ORDER],
                    "batch_app"
                )
                if report_path:
                    st.success("Batch drift report generated.")
                    with open(report_path, "rb") as f:
                        st.download_button("Download Batch Drift Report (HTML)", f,
                                           file_name=os.path.basename(report_path), key="batch_html")
                    with open(report_path, "r", encoding="utf-8") as f:
                        html = f.read()
                    components.html(html, height=600, scrolling=True)
                else:
                    st.info(f"Batch drift check skipped (need at least {MIN_DRIFT_ROWS} rows).")

                # Results table + download
                st.dataframe(results_df.sort_values('Default_Probability', ascending=False))
                st.download_button(
                    "Download Predictions CSV",
                    data=results_df.to_csv(index=False).encode("utf-8"),
                    file_name="loan_predictions.csv",
                    mime="text/csv"
                )

    # ---- Logs
    with tab3:
        st.subheader("Prediction Logs")
        log_file = "logs/predictions.log"
        if os.path.exists(log_file):
            with open(log_file, "r", encoding="utf-8") as f:
                st.text_area("Logs", f.read(), height=320)
            with open(log_file, "rb") as f:
                st.download_button("Download Prediction Logs", f, file_name="predictions.log")
        else:
            st.info("No logs yet. Make a prediction to start logging.")

if __name__ == "__main__":
    try:
        main()
    except Exception:
        st.error("Critical error occurred")
        st.code(traceback.format_exc())
        st.stop()

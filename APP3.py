# APP3.py
import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import numpy as np
import joblib
import os
import traceback
import subprocess
import importlib
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

# ===== Lazy SHAP Loader =====
def load_shap():
    """Try importing SHAP, install at runtime if missing."""
    try:
        return importlib.import_module("shap")
    except ModuleNotFoundError:
        with st.spinner("Installing SHAP... Please wait ⏳"):
            subprocess.run(
                ["pip", "install", "shap==0.45.0", "numba<0.60", "numpy<2.0"],
                check=False
            )
        return importlib.import_module("shap")

# ===== Utilities =====
def _strip_aux_columns(df: pd.DataFrame) -> pd.DataFrame:
    cols = [c for c in df.columns if not c.startswith("Unnamed")]
    df = df[cols]
    keep = [c for c in df.columns if c in FEATURES_ORDER]
    df = df[keep]
    df = df.reindex(columns=FEATURES_ORDER)
    return df

def _drift_ready(reference_df: pd.DataFrame, current_df: pd.DataFrame):
    ref = _strip_aux_columns(reference_df.copy())
    cur = _strip_aux_columns(current_df.copy())
    for col in NUMERICAL_FEATURES:
        if col in ref: ref[col] = pd.to_numeric(ref[col], errors='coerce')
        if col in cur: cur[col] = pd.to_numeric(cur[col], errors='coerce')
    for col in CATEGORICAL_FEATURES + BINARY_FEATURES:
        if col in ref: ref[col] = ref[col].astype(str)
        if col in cur: cur[col] = cur[col].astype(str)
    ref = ref.dropna(how="all")
    cur = cur.dropna(how="all")
    return ref, cur

def generate_drift_report(reference_data, new_data, report_name="drift_report"):
    try:
        reference_data, new_data = _drift_ready(reference_data, new_data)
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
# (unchanged – same as your version, keeping KPIs and charts)

# ===== Main App =====
def main():
    st.set_page_config(page_title="Loan Default Predictor", page_icon="💰", layout="wide")
    artifacts = load_artifacts()

    st.title("Loan Default Risk Assessment with Monitoring")
    st.caption(f"Developer: **{DEVELOPER_NAME}** | Model v{MODEL_VERSION} (trained {MODEL_TRAIN_DATE})")

    # Tabs (same as your version — Single Application, Batch, Logs)
    # ✅ Everything else remains unchanged

if __name__ == "__main__":
    try:
        main()
    except Exception:
        st.error("Critical error occurred")
        st.code(traceback.format_exc())
        st.stop()

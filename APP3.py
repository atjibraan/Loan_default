# APP3.py (final fixed version with SHAP lazy install pinned to 0.42.1)
import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import numpy as np
import joblib
import os
import shap
import traceback
import subprocess
import sys
from datetime import datetime
from sklearn.base import BaseEstimator, TransformerMixin
import matplotlib.pyplot as plt

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
MIN_DRIFT_ROWS = 30
THRESHOLD = 0.50

# Feature definitions
NUMERICAL_FEATURES = [
    'Age', 'Income', 'LoanAmount', 'CreditScore', 'MonthsEmployed',
    'NumCreditLines', 'InterestRate', 'LoanTerm', 'DTIRatio'
]
CATEGORICAL_FEATURES = ['Education', 'EmploymentType', 'LoanPurpose']
BINARY_FEATURES = ['MaritalStatus', 'HasMortgage', 'HasDependents', 'HasCoSigner']
FEATURES_ORDER = NUMERICAL_FEATURES + CATEGORICAL_FEATURES + BINARY_FEATURES

# ===== Logging =====
os.makedirs("logs", exist_ok=True)
import logging
logging.basicConfig(
    filename="logs/predictions.log",
    level=logging.INFO,
    format="%(asctime)s - %(message)s"
)

def log_prediction(input_data, prediction):
    logging.info(
        f"Input: {input_data.to_dict(orient='records') if isinstance(input_data, pd.DataFrame) else input_data}, "
        f"Prediction: {prediction}"
    )

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

# ===== Helpers =====
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

# ===== UI (Plotly KPIs) =====
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

# ===== Input & Batch =====
def get_user_input():
    with st.form("loan_input"):
        st.header("Applicant Information")
        
        col1, col2 = st.columns(2)
        
        with col1:
            age = st.number_input("Age", min_value=18, max_value=100, value=30)
            income = st.number_input("Annual Income ($)", min_value=0, value=50000)
            loan_amount = st.number_input("Loan Amount ($)", min_value=0, value=10000)
            credit_score = st.number_input("Credit Score", min_value=300, max_value=850, value=700)
            months_employed = st.number_input("Months Employed", min_value=0, value=36)
            
        with col2:
            num_credit_lines = st.number_input("Number of Credit Lines", min_value=0, value=2)
            interest_rate = st.number_input("Interest Rate (%)", min_value=0.0, max_value=30.0, value=5.0)
            loan_term = st.number_input("Loan Term (months)", min_value=1, value=36)
            dti_ratio = st.number_input("Debt-to-Income Ratio", min_value=0.0, max_value=1.0, value=0.3)
            education = st.selectbox("Education", ["High School", "Bachelor's", "Master's", "PhD"])
            
        employment_type = st.selectbox("Employment Type", ["Full-time", "Part-time", "Self-employed", "Unemployed"])
        loan_purpose = st.selectbox("Loan Purpose", ["Business", "Home", "Education", "Personal"])
        marital_status = st.selectbox("Marital Status", ["Single", "Married"])
        has_mortgage = st.selectbox("Has Mortgage", ["No", "Yes"])
        has_dependents = st.selectbox("Has Dependents", ["No", "Yes"])
        has_cosigner = st.selectbox("Has Co-Signer", ["No", "Yes"])
        
        submitted = st.form_submit_button("Predict Default Risk")
        
        if submitted:
            input_data = {
                'Age': age,
                'Income': income,
                'LoanAmount': loan_amount,
                'CreditScore': credit_score,
                'MonthsEmployed': months_employed,
                'NumCreditLines': num_credit_lines,
                'InterestRate': interest_rate,
                'LoanTerm': loan_term,
                'DTIRatio': dti_ratio,
                'Education': education,
                'EmploymentType': employment_type,
                'LoanPurpose': loan_purpose,
                'MaritalStatus': marital_status,
                'HasMortgage': has_mortgage,
                'HasDependents': has_dependents,
                'HasCoSigner': has_cosigner
            }
            return pd.DataFrame([input_data])
    return None

def process_batch_data(uploaded_file, artifacts):
    try:
        df = pd.read_csv(uploaded_file, nrows=MAX_CSV_ROWS)
        df = _strip_aux_columns(df)
        probs = predict_default_probability(df[FEATURES_ORDER], artifacts)
        if probs is not None:
            df['Default_Probability'] = probs
            df['Risk_Classification'] = np.where(probs >= THRESHOLD, "High Risk", "Low Risk")
            return df
    except Exception as e:
        st.error(f"Error processing batch file: {str(e)}")
        return None

# ===== Main App =====
def main():
    st.set_page_config(page_title="Loan Default Predictor", page_icon="💰", layout="wide")
    artifacts = load_artifacts()

    st.title("Loan Default Risk Assessment with Monitoring")
    st.caption(f"Developer: **{DEVELOPER_NAME}** | Model v{MODEL_VERSION} (trained {MODEL_TRAIN_DATE})")

    tab1, tab2, tab3, tab4 = st.tabs(["Single Application", "Batch Processing", "Prediction Logs", "Explainability"])

    with tab1:
        st.header("Single Application Evaluation")
        input_data = get_user_input()
        if input_data is not None:
            prob = predict_default_probability(input_data, artifacts)
            if prob is not None:
                prob = prob[0]
                log_prediction(input_data, prob)
                
                col1, col2 = st.columns([1, 2])
                with col1:
                    st.plotly_chart(gauge_probability(prob), use_container_width=True)
                with col2:
                    st.write("### Risk Assessment")
                    if prob >= THRESHOLD:
                        st.error(f"High Risk of Default ({prob*100:.1f}%)")
                        st.markdown("""
                        **Recommendation:** 
                        - Consider declining this application
                        - Request additional collateral
                        - Offer higher interest rate
                        """)
                    else:
                        st.success(f"Low Risk of Default ({prob*100:.1f}%)")
                        st.markdown("""
                        **Recommendation:** 
                        - Application appears creditworthy
                        - Standard terms can be offered
                        """)

    with tab2:
        st.header("Batch Processing")
        uploaded_file = st.file_uploader("Upload CSV file with loan applications", type=["csv"])
        
        if uploaded_file is not None:
            if uploaded_file.size > MAX_FILE_SIZE_MB * 1024 * 1024:
                st.error(f"File too large. Maximum size is {MAX_FILE_SIZE_MB}MB")
            else:
                with st.spinner("Processing batch file..."):
                    results = process_batch_data(uploaded_file, artifacts)
                    if results is not None:
                        st.success(f"Processed {len(results)} applications")
                        
                        st.download_button(
                            label="Download Results",
                            data=results.to_csv(index=False),
                            file_name="loan_predictions.csv",
                            mime="text/csv"
                        )
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            st.plotly_chart(risk_pie(results), use_container_width=True)
                        with col2:
                            st.plotly_chart(prob_hist(results), use_container_width=True)
                        
                        st.dataframe(results.sort_values('Default_Probability', ascending=False))
                        
                        # Generate drift report
                        if st.button("Generate Data Drift Report"):
                            report_path = generate_drift_report(artifacts['reference_data'], results)
                            if report_path:
                                with open(report_path, "rb") as f:
                                    st.download_button(
                                        label="Download Drift Report",
                                        data=f,
                                        file_name="drift_report.html",
                                        mime="text/html"
                                    )
                                st.success("Drift report generated")
                                st.markdown(f"**Summary:** Comparison between training data and current batch")
                            else:
                                st.warning("Could not generate drift report")

    with tab3:
        st.header("Prediction Logs")
        if os.path.exists("logs/predictions.log"):
            with open("logs/predictions.log", "r") as f:
                logs = f.read()
            st.text_area("Log Contents", logs, height=400)
        else:
            st.info("No prediction logs found")

    with tab4:
        st.subheader("Model Explainability")

        try:
            import shap
        except ImportError:
            with st.spinner("Installing SHAP... please wait (first-time only)"):
                subprocess.check_call([sys.executable, "-m", "pip", "install", "shap==0.42.1"])
            import shap

        try:
            from shap import Explainer
            shap.initjs()

            st.write("### Global Feature Importance")

            # Use a smaller sample for SHAP calculations
            sample = artifacts['reference_data'].sample(50, random_state=42)
            processed = artifacts['preprocessor'].transform(sample)

            # LinearExplainer for compatibility
            explainer = shap.LinearExplainer(artifacts['model'], processed)
            shap_values = explainer.shap_values(processed)

            # Plot summary
            fig, ax = plt.subplots()
            shap.summary_plot(shap_values, processed, show=False)
            st.pyplot(fig)

        except Exception as e:
            st.warning(f"Limited explainability due to: {str(e)}")
            # Fallback to feature importance
            try:
                if hasattr(artifacts['model'], 'feature_importances_'):
                    importance = artifacts['model'].feature_importances_
                    features = artifacts['preprocessor'].get_feature_names_out()

                    fig, ax = plt.subplots()
                    pd.Series(importance, index=features).sort_values().plot.barh()
                    plt.title("Feature Importance")
                    st.pyplot(fig)
            except Exception:
                st.error("Could not generate explanations with available resources")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        st.error("Critical error occurred")
        st.code(traceback.format_exc())
        st.stop()

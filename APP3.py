# APP3.py
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import os
import glob
import io
import traceback
from datetime import datetime
from sklearn.base import BaseEstimator, TransformerMixin

# Evidently for monitoring
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset

# ===== Configuration =====
MODEL_PATH = 'loan_default_model.pkl'
PREPROCESSOR_PATH = 'preprocessor.pkl'
REFERENCE_DATA_PATH = "X_train.csv"   # training features only (no target)
DEVELOPER_NAME = "Jibraan Attar"
MODEL_VERSION = "2.0"
MODEL_TRAIN_DATE = "2025-07-29"
MAX_CSV_ROWS = 2_000_000
MAX_FILE_SIZE_MB = 25

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
    """Log each prediction or batch summary."""
    try:
        if isinstance(input_data, pd.DataFrame):
            payload = input_data.to_dict(orient='records')
        else:
            payload = input_data
        logging.info(f"Input: {payload}, Prediction: {prediction}")
    except Exception as e:
        logging.error(f"Failed to log prediction: {e}")

# ===== Add Preprocessor Class (ensures joblib can unpickle) =====
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, OrdinalEncoder

class Preprocessor(BaseEstimator, TransformerMixin):
    """Custom Preprocessor used during training"""
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

# ===== File Guards =====
def guard_files():
    if not os.path.exists(MODEL_PATH):
        st.error(f"Model file not found at {MODEL_PATH}")
        st.stop()
    if not os.path.exists(PREPROCESSOR_PATH):
        st.error(f"Preprocessor file not found at {PREPROCESSOR_PATH}")
        st.stop()
    if not os.path.exists(REFERENCE_DATA_PATH):
        st.error(f"Reference training data not found at {REFERENCE_DATA_PATH} — ensure it contains only feature columns: {FEATURES_ORDER}")
        st.stop()

# ===== Helpers =====
@st.cache_resource(show_spinner=False)
def load_artifacts():
    artifacts = {
        'model': joblib.load(MODEL_PATH),
        'preprocessor': joblib.load(PREPROCESSOR_PATH),
        'reference_data_raw': pd.read_csv(REFERENCE_DATA_PATH)
    }
    # Clean reference data for drift (drop stray cols; ensure order)
    artifacts['reference_data'] = prepare_for_drift(artifacts['reference_data_raw'])
    return artifacts

def prepare_for_drift(df: pd.DataFrame) -> pd.DataFrame:
    """
    Align any dataset to the FEATURES_ORDER for Evidently drift:
    - Drop unexpected columns (e.g., LoanID, Unnamed: 0)
    - Add any missing expected columns (filled with NaN)
    - Reorder columns
    """
    df = df.copy()
    # Drop obvious index/id columns if present
    drop_candidates = [c for c in df.columns if c.lower().startswith('unnamed') or c.lower().endswith('id')]
    df = df.drop(columns=drop_candidates, errors='ignore')

    # Keep only expected features
    df = df[[c for c in FEATURES_ORDER if c in df.columns]]

    # Add missing columns (if any)
    for col in FEATURES_ORDER:
        if col not in df.columns:
            # Fill with sensible defaults: zeros for numeric, first option for categorical/binary
            if col in NUMERICAL_FEATURES:
                df[col] = np.nan  # let drift handle NaNs; we don't want to fabricate values
            elif col in CATEGORICAL_FEATURES:
                df[col] = pd.Series(dtype="object")
            else:  # binary
                df[col] = pd.Series(dtype="object")

    # Reorder strictly
    df = df[FEATURES_ORDER]

    # Coerce dtypes gently (numerics as numeric)
    for col in NUMERICAL_FEATURES:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    return df

def predict_default_probability(input_data: pd.DataFrame, artifacts):
    try:
        for col in NUMERICAL_FEATURES:
            input_data[col] = pd.to_numeric(input_data[col], errors='coerce')
        processed = artifacts['preprocessor'].transform(input_data)
        probs = artifacts['model'].predict_proba(processed)
        return probs[:, 1]
    except Exception as e:
        st.error(f"Prediction error: {str(e)}")
        with st.expander("Traceback"):
            st.code(traceback.format_exc())
        return None

def generate_drift_report(reference_data: pd.DataFrame, new_data: pd.DataFrame, report_name="drift_report"):
    """Generate and save Evidently HTML report; return file path or None."""
    try:
        ref = prepare_for_drift(reference_data)
        cur = prepare_for_drift(new_data)
        report = Report(metrics=[DataDriftPreset()])
        report.run(reference_data=ref, current_data=cur)
        report_path = f"logs/{report_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
        report.save_html(report_path)
        logging.info(f"Drift report saved: {report_path}")
        return report_path
    except Exception as e:
        msg = f"Error generating drift report: {str(e)}"
        logging.error(msg)
        st.warning(msg)
        return None

def get_user_input():
    with st.form("loan_input", border=True):
        st.subheader("Applicant Information")
        col1, col2, col3 = st.columns(3)
        age = col1.number_input("Age", 18, 100, 35)
        income = col2.number_input("Annual Income ($)", min_value=0, value=60000)
        loan_amount = col3.number_input("Loan Amount ($)", min_value=0, value=20000)

        col1, col2, col3 = st.columns(3)
        credit_score = col1.number_input("Credit Score", 300, 850, 700)
        months_employed = col2.number_input("Months Employed", min_value=0, value=36)
        num_credit_lines = col3.number_input("Number of Credit Lines", min_value=0, value=3)

        col1, col2, col3 = st.columns(3)
        interest_rate = col1.number_input("Interest Rate (%)", min_value=0.0, max_value=30.0, value=7.5, step=0.1)
        loan_term = col2.number_input("Loan Term (months)", min_value=1, value=36)
        dti_ratio = col3.number_input("DTI Ratio", min_value=0.0, max_value=1.0, value=0.35, step=0.01)

        st.subheader("Other Details")
        col1, col2, col3 = st.columns(3)
        education = col1.selectbox("Education", ["High School", "Bachelor's", "Master's", "PhD"])
        employment_type = col2.selectbox("Employment Type", ["Full-time", "Part-time", "Self-employed", "Unemployed"])
        loan_purpose = col3.selectbox("Loan Purpose", ["Business", "Home", "Education", "Auto"])

        col1, col2, col3, col4 = st.columns(4)
        marital_status = col1.selectbox("Marital Status", ["Single", "Married"])
        has_mortgage = col2.selectbox("Has Mortgage", ["No", "Yes"])
        has_dependents = col3.selectbox("Has Dependents", ["No", "Yes"])
        has_cosigner = col4.selectbox("Has Co-signer", ["No", "Yes"])

        col_a, col_b = st.columns([1, 1])
        predict_btn = col_a.form_submit_button("🔮 Predict Default Risk")
        clear_btn = col_b.form_submit_button("🧹 Clear")

        if predict_btn:
            return pd.DataFrame([{
                'Age': int(age),
                'Income': float(income),
                'LoanAmount': float(loan_amount),
                'CreditScore': int(credit_score),
                'MonthsEmployed': int(months_employed),
                'NumCreditLines': int(num_credit_lines),
                'InterestRate': float(interest_rate),
                'LoanTerm': int(loan_term),
                'DTIRatio': float(dti_ratio),
                'Education': education,
                'EmploymentType': employment_type,
                'MaritalStatus': marital_status,
                'HasMortgage': has_mortgage,
                'HasDependents': has_dependents,
                'LoanPurpose': loan_purpose,
                'HasCoSigner': has_cosigner
            }])
    return None

def create_gauge(prob_default: float):
    fig, ax = plt.subplots(figsize=(7, 1.4))
    ax.barh(['Risk'], [1], height=0.3, alpha=0.25)
    ax.barh(['Risk'], [prob_default], height=0.3)
    ax.set_xlim(0, 1)
    ax.set_xticks(np.linspace(0, 1, 6))
    ax.set_xticklabels([f'{x:.0%}' for x in np.linspace(0, 1, 6)])
    ax.set_yticks([])
    ax.set_title('Default Probability Gauge', pad=10)
    for spine in ["top", "right", "left"]:
        ax.spines[spine].set_visible(False)
    ax.grid(axis='x', linestyle='--', alpha=0.3)
    ax.text(prob_default, 0, f'{prob_default:.1%}', va='center', ha='center', fontsize=11, color='black')
    return fig

def create_prob_bars(prob_default: float):
    fig, ax = plt.subplots(figsize=(4.5, 2.6))
    ax.bar(['Default', 'No Default'], [prob_default, 1 - prob_default])
    ax.set_ylim(0, 1)
    ax.set_title('Probability Comparison', pad=10)
    ax.grid(axis='y', linestyle='--', alpha=0.3)
    return fig

def feature_importance_fig(model, feature_names):
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
        # Names after preprocessing are tricky; show top-level groups for intuition
        # Here, we approximate by repeating base names to match length if needed
        names = [f"f{i}" for i in range(len(importances))]
        fig, ax = plt.subplots(figsize=(6, 3))
        order = np.argsort(importances)[::-1][:15]
        ax.barh([names[i] for i in order][::-1], importances[order][::-1])
        ax.set_title("Model Feature Importances (top 15)")
        ax.grid(axis='x', linestyle='--', alpha=0.3)
        return fig
    return None

def read_prediction_log():
    path = "logs/predictions.log"
    if not os.path.exists(path):
        return pd.DataFrame(columns=["timestamp", "message"])
    lines = open(path, "r", encoding="utf-8", errors="ignore").read().splitlines()
    rows = []
    for ln in lines:
        # Format: "YYYY-mm-dd HH:MM:SS,ms - message"
        if " - " in ln:
            ts, msg = ln.split(" - ", 1)
            rows.append({"timestamp": ts, "message": msg})
        else:
            rows.append({"timestamp": "", "message": ln})
    return pd.DataFrame(rows)

def list_drift_reports():
    return sorted(glob.glob("logs/*_app_*.html") + glob.glob("logs/drift_report_*.html"))

# ===== Page =====
def main():
    st.set_page_config(page_title="Loan Default Predictor (Interactive MLOps)", page_icon="💰", layout="wide")
    guard_files()
    artifacts = load_artifacts()

    # Sidebar: model info & actions
    with st.sidebar:
        st.markdown("## 🧩 Model")
        st.markdown(f"**Developer:** {DEVELOPER_NAME}")
        st.markdown(f"**Version:** {MODEL_VERSION}")
        st.markdown(f"**Trained on:** {MODEL_TRAIN_DATE}")
        st.markdown("---")
        st.markdown("**Risk Rule:** High Risk if Probability ≥ 50%")

        st.markdown("---")
        st.markdown("### 🗂️ Files")
        st.caption(f"Model: `{MODEL_PATH}`")
        st.caption(f"Preprocessor: `{PREPROCESSOR_PATH}`")
        st.caption(f"Reference data: `{REFERENCE_DATA_PATH}`")

        st.markdown("---")
        st.markdown("### 🧪 Drift Reports")
        reports = list_drift_reports()
        if reports:
            choice = st.selectbox("Open a report", reports, index=len(reports)-1)
            with open(choice, "r", encoding="utf-8", errors="ignore") as f:
                html = f.read()
            st.markdown("Preview:")
            st.components.v1.html(html, height=350, scrolling=True)
            st.download_button("⬇️ Download this report", data=html, file_name=os.path.basename(choice), mime="text/html")
        else:
            st.info("No drift reports yet. Run a prediction or batch to generate one.")

        st.markdown("---")
        st.markdown("### 📜 Quick Logs Peek")
        df_log = read_prediction_log()
        if not df_log.empty:
            st.caption(df_log.tail(6).to_markdown(index=False))
        else:
            st.caption("_No logs yet_")

    # Header
    st.title("💰 Loan Default Risk — Interactive MLOps App")
    st.caption("Single predictions • Batch scoring • Drift monitoring • Prediction logs")

    # Tabs
    t1, t2, t3 = st.tabs(["🧍 Single Application", "📦 Batch Processing", "📑 Prediction Logs"])

    # -------- Single Application --------
    with t1:
        input_df = get_user_input()
        if input_df is not None:
            probs = predict_default_probability(input_df.copy(), artifacts)
            if probs is not None:
                prob_default = float(probs[0])
                prob_no_default = 1 - prob_default
                is_high_risk = prob_default >= 0.5
                prediction = "High Risk" if is_high_risk else "Low Risk"

                m1, m2, m3 = st.columns(3)
                m1.metric("Probability of Default", f"{prob_default:.2%}")
                m2.metric("Probability of No Default", f"{prob_no_default:.2%}")
                m3.metric("Risk Classification", prediction)

                c1, c2 = st.columns([2, 1.3])
                with c1:
                    st.pyplot(create_gauge(prob_default), use_container_width=True)
                with c2:
                    st.pyplot(create_prob_bars(prob_default), use_container_width=True)

                # Optional feature importance
                fi_fig = feature_importance_fig(artifacts['model'], FEATURES_ORDER)
                if fi_fig is not None:
                    st.subheader("🔍 Model Feature Importances (if available)")
                    st.pyplot(fi_fig, use_container_width=True)

                # Recommendations
                if is_high_risk:
                    st.error("⚠️ This case is classified as **HIGH RISK**.")
                    st.markdown(
                        "- Request additional financial docs\n"
                        "- Consider a co-signer requirement\n"
                        "- Price the risk (higher interest) or decline if above policy"
                    )
                else:
                    st.success("✅ This case is classified as **LOW RISK**.")
                    st.markdown(
                        "- Consider approval at standard rates\n"
                        "- May qualify for preferred products"
                    )

                # Logging + Drift
                log_prediction(input_df, prediction)
                if st.toggle("Generate drift report for this input", value=True, help="Compares current input vs. training features"):
                    report_path = generate_drift_report(artifacts['reference_data'], input_df, "single_app")
                    if report_path:
                        with open(report_path, "r", encoding="utf-8", errors="ignore") as f:
                            html = f.read()
                        st.components.v1.html(html, height=450, scrolling=True)
                        st.download_button("⬇️ Download Drift Report", html, file_name=os.path.basename(report_path), mime="text/html")

    # -------- Batch Processing --------
    with t2:
        st.info(f"Upload CSV with required columns. Max size **{MAX_FILE_SIZE_MB}MB**, rows up to **{MAX_CSV_ROWS:,}**.")
        uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
        if uploaded_file is not None:
            file_size = len(uploaded_file.getvalue()) / (1024 * 1024)
            st.caption(f"Uploaded: **{uploaded_file.name}** ({file_size:.2f} MB)")

            # Read & validate subset of columns
            try:
                df_raw = pd.read_csv(uploaded_file, nrows=MAX_CSV_ROWS)
            except Exception as e:
                st.error(f"Failed to read CSV: {e}")
                st.stop()

            # Align to FEATURES_ORDER; warn if columns missing
            df_for_pred = df_raw.copy()
            missing = [c for c in FEATURES_ORDER if c not in df_for_pred.columns]
            if missing:
                st.warning(f"Missing columns will be added as empty (NaN) for processing: {missing}")
            for col in FEATURES_ORDER:
                if col not in df_for_pred.columns:
                    df_for_pred[col] = np.nan
            df_for_pred = df_for_pred[FEATURES_ORDER]

            probs = predict_default_probability(df_for_pred.copy(), artifacts)
            if probs is not None:
                results = df_raw.copy()
                results['Default_Probability'] = probs

                colL, colR = st.columns([1, 3])
                with colL:
                    threshold = st.slider("Risk Threshold", min_value=0.1, max_value=0.9, value=0.5, step=0.05,
                                          help="Classify as High Risk if probability ≥ threshold")
                results['Risk_Classification'] = np.where(results['Default_Probability'] >= threshold, "High Risk", "Low Risk")

                # KPIs
                k1, k2, k3 = st.columns(3)
                k1.metric("Total Rows", len(results))
                k2.metric("High Risk", int((results['Risk_Classification'] == "High Risk").sum()))
                k3.metric("Avg Default Prob", f"{results['Default_Probability'].mean():.2%}")

                # Histogram of probabilities
                fig, ax = plt.subplots(figsize=(6.5, 3))
                ax.hist(results['Default_Probability'], bins=30)
                ax.axvline(threshold, linestyle='--')
                ax.set_title("Distribution of Predicted Default Probabilities")
                ax.set_xlabel("Probability")
                ax.set_ylabel("Count")
                ax.grid(axis='y', linestyle='--', alpha=0.3)
                st.pyplot(fig, use_container_width=True)

                # Tables
                hi_df = results[results['Risk_Classification'] == "High Risk"].copy()
                lo_df = results[results['Risk_Classification'] == "Low Risk"].copy()

                t_hi, t_lo = st.tabs([
                    f"🔺 High Risk ({len(hi_df)})",
                    f"🔻 Low Risk ({len(lo_df)})"
                ])
                with t_hi:
                    if not hi_df.empty:
                        st.dataframe(hi_df.sort_values('Default_Probability', ascending=False), use_container_width=True)
                    else:
                        st.info("No high risk rows at this threshold.")
                with t_lo:
                    if not lo_df.empty:
                        st.dataframe(lo_df.sort_values('Default_Probability', ascending=True), use_container_width=True)
                    else:
                        st.info("No low risk rows at this threshold.")

                # Downloads
                csv_bytes = results.to_csv(index=False).encode('utf-8')
                st.download_button("⬇️ Download Scored CSV", data=csv_bytes, file_name="loan_predictions_scored.csv", mime="text/csv")

                # Log batch & Drift report
                log_prediction(df_for_pred.head(5), f"Batch scored: {len(results)} rows")  # log sample to avoid huge logs
                if st.toggle("Generate batch drift report", value=True, help="Compare uploaded data vs. training features"):
                    report_path = generate_drift_report(artifacts['reference_data'], df_for_pred, "batch_app")
                    if report_path:
                        with open(report_path, "r", encoding="utf-8", errors="ignore") as f:
                            html = f.read()
                        st.components.v1.html(html, height=450, scrolling=True)
                        st.download_button("⬇️ Download Batch Drift Report", html, file_name=os.path.basename(report_path), mime="text/html")

    # -------- Prediction Logs --------
    with t3:
        st.subheader("📑 Prediction Logs")
        st.caption("These are the entries written to `logs/predictions.log`.")

        df_log = read_prediction_log()
        if df_log.empty:
            st.info("No logs yet.")
        else:
            c1, c2 = st.columns([1, 3])
            with c1:
                search = st.text_input("Search text")
            with c2:
                n_rows = st.slider("Rows to show", min_value=5, max_value=1000, value=200, step=5)

            if search:
                m = df_log['message'].str.contains(search, case=False, na=False) | df_log['timestamp'].str.contains(search, case=False, na=False)
                view = df_log[m].tail(n_rows)
            else:
                view = df_log.tail(n_rows)

            st.dataframe(view, use_container_width=True, height=420)

            # Download logs
            st.download_button("⬇️ Download Full Log", data=open("logs/predictions.log", "rb").read(),
                               file_name="predictions.log", mime="text/plain")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        st.error("Critical error occurred")
        st.code(traceback.format_exc())
        st.stop()

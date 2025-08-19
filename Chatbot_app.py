# loan_default_predictor.py
import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import numpy as np
import joblib
import os
import io
import base64
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
MAX_CSV_ROWS = 20000000000
MAX_FILE_SIZE_MB = 25
MIN_DRIFT_ROWS = 30       # Skip drift if fewer rows than this
THRESHOLD = 0.50          # Risk cut-off

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

# ===== Explainability helpers (SHAP-free) =====
def get_transformed_feature_names(preprocessor: Preprocessor):
    try:
        return list(preprocessor.column_transformer.get_feature_names_out())
    except Exception:
        # Fallback: basic names
        names = []
        names += [f"num__{c}" for c in NUMERICAL_FEATURES]
        # For categorical, OneHotEncoder(drop='first') -> k-1 columns, but names unknown here
        names += ["cat__"] * len(CATEGORICAL_FEATURES)
        names += [f"bin__{c}" for c in BINARY_FEATURES]
        return names

def plot_model_feature_importance(artifacts, top_n=20):
    """Use model-native feature_importances_ if available; else show a notice."""
    model = artifacts['model']
    pre = artifacts['preprocessor']
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
        feat_names = get_transformed_feature_names(pre)
        # Align length if needed
        n = min(len(importances), len(feat_names))
        s = pd.Series(importances[:n], index=feat_names[:n]).sort_values(ascending=False).head(top_n)
        fig = px.bar(s[::-1], orientation='h', title="Global Feature Importance (Model-based)",
                     labels={"value": "Importance", "index": "Feature"})
        fig.update_layout(height=500, margin=dict(l=10, r=10, t=40, b=10))
        return fig
    else:
        fig = go.Figure()
        fig.add_annotation(text="Model does not expose feature_importances_.", showarrow=False, x=0.5, y=0.5)
        fig.update_layout(height=300)
        return fig

def suggest_counterfactuals(input_df: pd.DataFrame, artifacts, threshold=THRESHOLD, max_iters=20):
    """
    Simple heuristic search: nudge numeric features in directions likely to reduce default risk.
    Returns suggestions (list of dict) and the best adjusted row (DataFrame of 1 row).
    """
    row = input_df.iloc[0].copy()
    best_row = row.copy()
    best_prob = float(predict_default_probability(pd.DataFrame([row]), artifacts)[0])

    # Define directions: +1 to increase, -1 to decrease, 0 ignore
    directions = {
        "CreditScore": +1,
        "Income": +1,
        "MonthsEmployed": +1,
        "LoanAmount": -1,
        "InterestRate": -1,
        "DTIRatio": -1,
        "NumCreditLines": -1,  # assume fewer lines may help
        "Age": 0,
        "LoanTerm": 0,  # ambiguous (longer term can reduce payment burden but raise risk)
    }
    # Step sizes (fraction of current or absolute)
    frac_step = {
        "Income": 0.10,
        "LoanAmount": 0.10,
        "DTIRatio": 0.05,
        "InterestRate": 0.10,
    }
    abs_step = {
        "CreditScore": 10,
        "MonthsEmployed": 6,
        "NumCreditLines": 1,
    }

    suggestions = []
    for _ in range(max_iters):
        improved = False
        for feat, direction in directions.items():
            if direction == 0 or feat not in row:
                continue
            candidate = row.copy()
            if feat in frac_step:
                delta = max(1e-6, candidate[feat] * frac_step[feat])
            else:
                delta = abs_step.get(feat, 1.0)

            candidate[feat] = candidate[feat] + direction * delta
            # Bounds
            if feat == "CreditScore":
                candidate[feat] = float(np.clip(candidate[feat], 300, 850))
            if feat == "DTIRatio":
                candidate[feat] = float(np.clip(candidate[feat], 0.0, 2.0))
            if feat == "InterestRate":
                candidate[feat] = float(max(0.0, candidate[feat]))

            prob = float(predict_default_probability(pd.DataFrame([candidate]), artifacts)[0])
            if prob < best_prob:
                change = candidate[feat] - row[feat]
                suggestions.append({
                    "feature": feat,
                    "change": change,
                    "from": row[feat],
                    "to": candidate[feat],
                    "new_prob": prob
                })
                best_prob = prob
                best_row = candidate.copy()
                row = candidate.copy()
                improved = True
                if best_prob < threshold:
                    return suggestions, pd.DataFrame([best_row])
        if not improved:
            break
    return suggestions, pd.DataFrame([best_row])

def what_if_prediction(base_df: pd.DataFrame, artifacts, overrides: dict):
    """Apply overrides to base_df (single row) and return new prob."""
    modified = base_df.copy()
    for k, v in overrides.items():
        if k in modified.columns:
            modified.loc[modified.index[0], k] = v
    prob = float(predict_default_probability(modified, artifacts)[0])
    return prob, modified

def make_risk_heatmap(artifacts, base_row: pd.Series,
                      x_feat="CreditScore", y_feat="DTIRatio",
                      x_range=(300, 850), y_range=(0.0, 1.5), x_steps=40, y_steps=40):
    """
    Score a grid varying two features; hold others at base_row.
    """
    xs = np.linspace(x_range[0], x_range[1], x_steps)
    ys = np.linspace(y_range[0], y_range[1], y_steps)
    grid = []
    for y in ys:
        row_list = []
        for x in xs:
            r = base_row.copy()
            r[x_feat] = x
            r[y_feat] = y
            prob = float(predict_default_probability(pd.DataFrame([r]), artifacts)[0])
            row_list.append(prob)
        grid.append(row_list)
    z = np.array(grid)
    fig = go.Figure(data=go.Heatmap(
        z=z, x=xs, y=ys, colorscale="RdYlGn_r", colorbar_title="Default Prob"
    ))
    fig.add_contour(z=z, x=xs, y=ys, showscale=False, contours_coloring='lines', line_width=1)
    fig.update_layout(
        title=f"Risk Heatmap: {x_feat} vs {y_feat}",
        xaxis_title=x_feat, yaxis_title=y_feat,
        height=500, margin=dict(l=10, r=10, t=40, b=10)
    )
    # mark applicant
    fig.add_trace(go.Scatter(
        x=[base_row[x_feat]], y=[base_row[y_feat]],
        mode="markers", marker=dict(size=10, line=dict(width=1, color="black")),
        name="Applicant"
    ))
    return fig

def make_radar_chart(artifacts, input_row: pd.Series, threshold=THRESHOLD):
    """
    Compare applicant vs. pseudo cohorts derived from reference_data using model predictions:
    - Defaulters: prob >= threshold
    - Non-defaulters: prob < threshold
    Normalize numeric features to 0-1 based on reference_data min/max.
    """
    ref = artifacts['reference_data'].copy()
    # Need to predict probabilities on ref to build cohorts
    probs_ref = predict_default_probability(ref.copy(), artifacts)
    if probs_ref is None:
        fig = go.Figure()
        fig.add_annotation(text="Unable to compute radar chart.", showarrow=False, x=0.5, y=0.5)
        return fig
    ref = ref.assign(prob=probs_ref)
    def_cohort = ref[ref['prob'] >= threshold]
    nondef_cohort = ref[ref['prob'] < threshold]

    # Normalize numeric features
    cats = CATEGORICAL_FEATURES + BINARY_FEATURES
    num = NUMERICAL_FEATURES
    mins = ref[num].min()
    maxs = ref[num].max().replace(0, 1)  # avoid zero range
    def_mean = def_cohort[num].mean()
    nondef_mean = nondef_cohort[num].mean()

    def norm(s):
        return (s - mins) / (maxs - mins + 1e-9)

    applicant_norm = norm(input_row[num])
    def_norm = norm(def_mean)
    nondef_norm = norm(nondef_mean)

    categories = num + [num[0]]  # close the loop
    applicant_vals = list(applicant_norm.values) + [applicant_norm.values[0]]
    def_vals = list(def_norm.values) + [def_norm.values[0]]
    nondef_vals = list(nondef_norm.values) + [nondef_norm.values[0]]

    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(r=nondef_vals, theta=categories, fill='toself', name='Avg Non-Defaulters'))
    fig.add_trace(go.Scatterpolar(r=def_vals, theta=categories, fill='toself', name='Avg Defaulters'))
    fig.add_trace(go.Scatterpolar(r=applicant_vals, theta=categories, fill='toself', name='Applicant'))
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
        showlegend=True,
        title="Feature Radar: Applicant vs Cohorts (normalized)",
        height=500, margin=dict(l=10, r=10, t=40, b=10)
    )
    return fig

def generate_html_report(input_df: pd.DataFrame, prob: float, classification: str,
                         recommendations: list, suggestions: list, save_path: str):
    """
    Create a lightweight HTML report (no external deps) and save to disk.
    """
    row = input_df.iloc[0].to_dict()
    rec_html = "".join([f"<li>{r}</li>" for r in recommendations])
    sug_html = "".join([f"<li><b>{s['feature']}</b>: {s['from']} → {s['to']} (new p={s['new_prob']:.2%})</li>" for s in suggestions])
    table_rows = "".join([f"<tr><td>{k}</td><td>{v}</td></tr>" for k, v in row.items()])

    html = f"""
    <html>
    <head>
        <meta charset="utf-8"/>
        <title>Loan Default Risk Report</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 24px; }}
            h1, h2 {{ color: #333; }}
            table {{ border-collapse: collapse; width: 100%; }}
            th, td {{ border: 1px solid #ddd; padding: 8px; }}
            th {{ background: #f6f6f6; text-align: left; }}
            .kpi {{ display: flex; gap: 24px; margin: 12px 0 24px; }}
            .badge {{ padding: 6px 10px; border-radius: 6px; color: white; }}
            .badge.ok {{ background: #2ca02c; }}
            .badge.bad {{ background: #d62728; }}
        </style>
    </head>
    <body>
        <h1>Loan Default Risk Report</h1>
        <p><b>Generated:</b> {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>
        <div class="kpi">
            <div><b>Default Probability:</b> {prob:.2%}</div>
            <div><b>Risk Classification:</b> <span class="badge {'bad' if classification=='High Risk' else 'ok'}">{classification}</span></div>
            <div><b>Threshold:</b> {THRESHOLD:.0%}</div>
        </div>
        <h2>Applicant Inputs</h2>
        <table>
            <thead><tr><th>Feature</th><th>Value</th></tr></thead>
            <tbody>{table_rows}</tbody>
        </table>
        <h2>Recommendations</h2>
        <ul>{rec_html}</ul>
        <h2>Counterfactual Suggestions</h2>
        <ul>{sug_html if sug_html else "<i>No actionable changes found within search budget.</i>"}</ul>
        <p style="margin-top:24px; font-size:12px; color:#666;">
            Developer: {DEVELOPER_NAME} • Model v{MODEL_VERSION} (trained {MODEL_TRAIN_DATE})
        </p>
    </body>
    </html>
    """
    with open(save_path, "w", encoding="utf-8") as f:
        f.write(html)
    return save_path

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

def plot_gauge_simple(prob):
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=float(prob),
        title={'text': "Default Probability"},
        gauge={'axis': {'range':[0,1]}, 'bar': {'color':'orange'},
               'steps':[{'range':[0,0.5],'color':'lightgreen'},{'range':[0.5,1],'color':'red'}]}
    ))
    fig.update_layout(height=300)
    return fig

def plot_probability_bar(prob):
    fig = go.Figure(go.Bar(
        x=['Default','No Default'],
        y=[float(prob),1-float(prob)],
        marker_color=['#ff7f0e','#1f77b4']
    ))
    fig.update_layout(yaxis=dict(range=[0,1]), title='Probability Comparison')
    return fig

def risk_pie(df):
    counts = df['Risk_Classification'].value_counts().reset_index()
    counts.columns = ['Risk', 'Count']
    fig = px.pie(counts, names='Risk', values='Count', hole=0.4, title='Risk Split')
    fig.update_layout(height=330, margin=dict(l=10, r=10, t=40, b=10))
    return fig

def plot_risk_pie_simple(df):
    risk_counts = df['Risk_Classification'].value_counts()
    fig = go.Figure(go.Pie(
        labels=risk_counts.index,
        values=risk_counts.values,
        hole=0.4,
        marker_colors=['#ff7f0e','#1f77b4']
    ))
    fig.update_layout(title="Risk Distribution")
    return fig

def prob_hist(df):
    fig = px.histogram(df, x='Default_Probability', nbins=30,
                       title='Predicted Probability Distribution')
    fig.update_layout(height=330, xaxis_title="Default Probability",
                      yaxis_title="Count", margin=dict(l=10, r=10, t=40, b=10))
    return fig

# ===== Suggestion Engine =====
def generate_suggestions(data, prob):
    tips = []
    contributions = []
    if data['Income'] < 30000:
        tips.append("💡 Consider increasing your income or providing a co-signer.")
        contributions.append("low Income")
    if data['DTIRatio'] > 0.4:
        tips.append("💡 Your DTI is high; consider reducing debt or negotiating interest rate.")
        contributions.append("high DTI")
    if data['LoanAmount'] > data['Income']*5:
        tips.append("💡 Your LoanAmount is high relative to income; reduce loan size or improve income.")
        contributions.append("high LoanAmount")
    if contributions:
        contribution_text = ' and '.join(contributions)
        tips.insert(0,f"⚠️ Key factors contributing to risk: {contribution_text}")
    return tips

# ===== Helpers: Recommendations text =====
def recommendations_for_prob(prob: float):
    if prob >= THRESHOLD:
        return [
            "Consider declining this application.",
            "Request additional collateral or a guarantor.",
            "Offer adjusted terms (e.g., higher interest, lower amount).",
            "Ask for documentation to verify income and obligations."
        ]
    else:
        return [
            "Application appears creditworthy on standard terms.",
            "Optionally consider slightly better terms for retention.",
            "Continue monitoring via periodic risk checks."
        ]

# ===== Forms and Batch =====
def get_user_input():
    with st.form("loan_input"):
        st.header("Applicant Information")
        col1, col2, col3 = st.columns(3)
        age = col1.number_input("Age", 18, 100, 35)
        income = col2.number_input("Annual Income ($)", 1000, 500000000, 60000)
        loan_amount = col3.number_input("Loan Amount ($)", 500, 200000000, 200000)

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

# ===== Chatbot Interface =====
def chatbot_interface(artifacts):
    st.subheader("Multi-turn Chatbot")
    if 'conversation' not in st.session_state:
        st.session_state.conversation = []
        st.session_state.user_data = {}

    def display_chat():
        for entry in st.session_state.conversation:
            if entry['sender']=='bot':
                st.markdown(f"**Bot:** {entry['message']}")
            else:
                st.markdown(f"**You:** {entry['message']}")
    display_chat()

    # Group fields into categories for better organization
    personal_info = [
        ("Age","number"), 
        ("Income","number"),
        ("MaritalStatus","select",["Single","Married"]),
        ("HasDependents","select",["No","Yes"])
    ]
    
    loan_info = [
        ("LoanAmount","number"),
        ("LoanTerm","number"),
        ("LoanPurpose","select",["Business","Home","Education","Auto"]),
        ("InterestRate","number")
    ]
    
    financial_info = [
        ("CreditScore","number"),
        ("NumCreditLines","number"),
        ("DTIRatio","number"),
        ("HasMortgage","select",["No","Yes"])
    ]
    
    employment_info = [
        ("MonthsEmployed","number"),
        ("EmploymentType","select",["Full-time","Part-time","Self-employed","Unemployed"]),
        ("Education","select",["High School","Bachelor's","Master's","PhD"]),
        ("HasCoSigner","select",["No","Yes"])
    ]

    # Check which step we're on
    all_steps = personal_info + loan_info + financial_info + employment_info
    for field in all_steps:
        if field[0] not in st.session_state.user_data:
            st.session_state.current_step = field
            break
    else:
        st.session_state.current_step = None

    if st.session_state.current_step:
        field = st.session_state.current_step
        st.markdown(f"**Bot:** Please provide your {field[0]}")
        
        # Create a form for each category
        with st.form(key='input_form'):
            cols = st.columns(4)
            
            # Personal Information
            with cols[0]:
                st.subheader("Personal Info")
                for f in personal_info:
                    if f[1] == "number":
                        value = st.number_input(f[0], value=0, key=f"input_{f[0]}")
                    else:
                        value = st.selectbox(f[0], options=f[2], key=f"input_{f[0]}")
                    if f[0] not in st.session_state.user_data:
                        st.session_state.user_data[f[0]] = value
            
            # Loan Information
            with cols[1]:
                st.subheader("Loan Info")
                for f in loan_info:
                    if f[1] == "number":
                        value = st.number_input(f[0], value=0, key=f"input_{f[0]}")
                    else:
                        value = st.selectbox(f[0], options=f[2], key=f"input_{f[0]}")
                    if f[0] not in st.session_state.user_data:
                        st.session_state.user_data[f[0]] = value
            
            # Financial Information
            with cols[2]:
                st.subheader("Financial Info")
                for f in financial_info:
                    if f[1] == "number":
                        value = st.number_input(f[0], value=0, key=f"input_{f[0]}")
                    else:
                        value = st.selectbox(f[0], options=f[2], key=f"input_{f[0]}")
                    if f[0] not in st.session_state.user_data:
                        st.session_state.user_data[f[0]] = value
            
            # Employment Information
            with cols[3]:
                st.subheader("Employment Info")
                for f in employment_info:
                    if f[1] == "number":
                        value = st.number_input(f[0], value=0, key=f"input_{f[0]}")
                    else:
                        value = st.selectbox(f[0], options=f[2], key=f"input_{f[0]}")
                    if f[0] not in st.session_state.user_data:
                        st.session_state.user_data[f[0]] = value
            
            if st.form_submit_button("Submit All Information"):
                for field in all_steps:
                    st.session_state.conversation.append({
                        'sender': 'user',
                        'message': f"{field[0]}: {st.session_state.user_data[field[0]]}"
                    })
                st.session_state.conversation.append({
                    'sender': 'bot',
                    'message': "All information recorded. Calculating your results..."
                })
                st.experimental_rerun()
    else:
        st.markdown("**Bot:** Thank you! Here's your loan default dashboard...")
        df_input = pd.DataFrame([st.session_state.user_data])
        probs = predict_default_probability(df_input, artifacts)
        if probs is not None:
            p = float(probs[0])
            prediction = "High Risk" if p >= THRESHOLD else "Low Risk"

            # KPI row
            c1, c2 = st.columns(2)
            c1.metric("Default Probability", f"{p:.2%}")
            c2.metric("Risk Classification", prediction)
            
            st.plotly_chart(plot_gauge_simple(p), use_container_width=True)
            st.plotly_chart(plot_probability_bar(p), use_container_width=True)

            # Generate actionable suggestions
            tips = generate_suggestions(st.session_state.user_data, p)
            for tip in tips:
                if "⚠️" in tip:
                    st.warning(tip)
                else:
                    st.info(tip)

            st.subheader("Applicant Overview")
            st.dataframe(df_input)

            # Log
            log_prediction(df_input, {"class": prediction, "prob": p})

            # Visual storytelling
            st.markdown("### 🎨 Visual Storytelling")
            base_row = df_input.iloc[0]
            hm = make_risk_heatmap(artifacts, base_row, "CreditScore", "DTIRatio",
                                   x_range=(300, 850), y_range=(0.0, 1.5))
            st.plotly_chart(hm, use_container_width=True)
            radar = make_radar_chart(artifacts, base_row)
            st.plotly_chart(radar, use_container_width=True)

            # Counterfactuals
            st.markdown("### 💡 Counterfactual Suggestions")
            suggestions, best_df = suggest_counterfactuals(df_input.copy(), artifacts, THRESHOLD)
            if len(suggestions) == 0:
                st.info("No actionable single-step changes found within the search budget.")
            else:
                s_df = pd.DataFrame(suggestions)
                s_df['new_prob'] = s_df['new_prob'].map(lambda x: f"{x:.2%}")
                st.dataframe(s_df, use_container_width=True)

            # What-if tool
            st.markdown("### 🧪 What-If Analysis")
            with st.expander("Adjust key features and see impact"):
                colA, colB, colC, colD = st.columns(4)
                w_credit = colA.slider("CreditScore", 300, 850, int(base_row['CreditScore']))
                w_income = colB.number_input("Income ($)", min_value=0, value=int(base_row['Income']), step=1000)
                w_amount = colC.number_input("LoanAmount ($)", min_value=0, value=int(base_row['LoanAmount']), step=1000)
                w_dti = colD.slider("DTIRatio", 0.0, 2.0, float(base_row['DTIRatio']), 0.01)

                overrides = {
                    "CreditScore": w_credit,
                    "Income": w_income,
                    "LoanAmount": w_amount,
                    "DTIRatio": w_dti
                }
                new_prob, mod_df = what_if_prediction(df_input.copy(), artifacts, overrides)
                new_pred = "High Risk" if new_prob >= THRESHOLD else "Low Risk"

                cc1, cc2, cc3 = st.columns(3)
                cc1.metric("New Default Prob", f"{new_prob:.2%}", delta=f"{(new_prob - p):+.2%}")
                cc2.metric("New Class", new_pred)
                cc3.metric("Threshold", f"{THRESHOLD:.0%}")
                st.plotly_chart(plot_gauge_simple(new_prob), use_container_width=True)

            # Recommendations & Report
            st.markdown("### 📄 Recommendations & Downloadable Report")
            recs = recommendations_for_prob(p)
            st.write("- " + "\n- ".join(recs))
            if st.button("Generate Decision Report (HTML)"):
                save_to = f"logs/decision_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
                path = generate_html_report(df_input.copy(), p, prediction, recs, suggestions, save_path=save_to)
                with open(path, "rb") as f:
                    st.download_button("Download Report (HTML)", f, file_name=os.path.basename(path), mime="text/html")

# ===== Main App =====
def main():
    st.set_page_config(page_title="Loan Default Predictor", page_icon="💰", layout="wide")
    artifacts = load_artifacts()

    st.title("Loan Default Risk Assessment with Monitoring")
    st.caption(f"Developer: **{DEVELOPER_NAME}** | Model v{MODEL_VERSION} (trained {MODEL_TRAIN_DATE})")

    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "Single Application", "Chatbot Interface", "Batch Processing", "Prediction Logs", "Explainability"
    ])

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
                log_prediction(input_df, {"class": prediction, "prob": p})

                # Visual storytelling
                st.markdown("### 🎨 Visual Storytelling")
                base_row = input_df.iloc[0]
                hm = make_risk_heatmap(artifacts, base_row, "CreditScore", "DTIRatio",
                                       x_range=(300, 850), y_range=(0.0, 1.5))
                st.plotly_chart(hm, use_container_width=True)
                radar = make_radar_chart(artifacts, base_row)
                st.plotly_chart(radar, use_container_width=True)

                # Counterfactuals
                st.markdown("### 💡 Counterfactual Suggestions")
                suggestions, best_df = suggest_counterfactuals(input_df.copy(), artifacts, THRESHOLD)
                if len(suggestions) == 0:
                    st.info("No actionable single-step changes found within the search budget.")
                else:
                    s_df = pd.DataFrame(suggestions)
                    s_df['new_prob'] = s_df['new_prob'].map(lambda x: f"{x:.2%}")
                    st.dataframe(s_df, use_container_width=True)

                # What-if tool
                st.markdown("### 🧪 What-If Analysis")
                with st.expander("Adjust key features and see impact"):
                    colA, colB, colC, colD = st.columns(4)
                    w_credit = colA.slider("CreditScore", 300, 850, int(base_row['CreditScore']))
                    w_income = colB.number_input("Income ($)", min_value=0, value=int(base_row['Income']), step=1000)
                    w_amount = colC.number_input("LoanAmount ($)", min_value=0, value=int(base_row['LoanAmount']), step=1000)
                    w_dti = colD.slider("DTIRatio", 0.0, 2.0, float(base_row['DTIRatio']), 0.01)

                    overrides = {
                        "CreditScore": w_credit,
                        "Income": w_income,
                        "LoanAmount": w_amount,
                        "DTIRatio": w_dti
                    }
                    new_prob, mod_df = what_if_prediction(input_df.copy(), artifacts, overrides)
                    new_pred = "High Risk" if new_prob >= THRESHOLD else "Low Risk"

                    cc1, cc2, cc3 = st.columns(3)
                    cc1.metric("New Default Prob", f"{new_prob:.2%}", delta=f"{(new_prob - p):+.2%}")
                    cc2.metric("New Class", new_pred)
                    cc3.metric("Threshold", f"{THRESHOLD:.0%}")
                    st.plotly_chart(gauge_probability(new_prob), use_container_width=True)

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

                # Recommendations & Report
                st.markdown("### 📄 Recommendations & Downloadable Report")
                recs = recommendations_for_prob(p)
                st.write("- " + "\n- ".join(recs))
                if st.button("Generate Decision Report (HTML)"):
                    save_to = f"logs/decision_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
                    path = generate_html_report(input_df.copy(), p, prediction, recs, suggestions, save_path=save_to)
                    with open(path, "rb") as f:
                        st.download_button("Download Report (HTML)", f, file_name=os.path.basename(path), mime="text/html")

    # ---- Chatbot Interface
    with tab2:
        chatbot_interface(artifacts)

    # ---- Batch Processing
    with tab3:
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
    with tab4:
        st.subheader("Prediction Logs")
        log_file = "logs/predictions.log"
        if os.path.exists(log_file):
            with open(log_file, "r", encoding="utf-8") as f:
                st.text_area("Logs", f.read(), height=320)
            with open(log_file, "rb") as f:
                st.download_button("Download Prediction Logs", f, file_name="predictions.log")
        else:
            st.info("No logs yet. Make a prediction to start logging.")

    # ---- Explainability (Global + Visuals in one place too)
    with tab5:
        st.subheader("Global Explainability (No SHAP)")
        st.plotly_chart(plot_model_feature_importance(artifacts), use_container_width=True)
        st.caption("Tip: Choose two influential features and inspect the Risk Heatmap in the Single Application tab.")

if __name__ == "__main__":
    try:
        main()
    except Exception:
        st.error("Critical error occurred")
        st.code(traceback.format_exc())
        st.stop()

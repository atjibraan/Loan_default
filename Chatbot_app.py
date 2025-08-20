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
import plotly.figure_factory as ff

# Evidently for monitoring
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset, TargetDriftPreset
from evidently.metrics import *
from evidently.pipeline.column_mapping import ColumnMapping

# ===== Configuration =====
MODEL_PATH = 'loan_default_model.pkl'
PREPROCESSOR_PATH = 'preprocessor.pkl'
REFERENCE_DATA_PATH = "X_train.csv"
TARGET_DATA_PATH = "y_train.csv"
DEVELOPER_NAME = "Jibraan Attar"
MODEL_VERSION = "2.0"
MODEL_TRAIN_DATE = "2025-07-29"
MAX_CSV_ROWS = 20000000000
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

# ===== Logging Setup =====
os.makedirs("logs", exist_ok=True)
os.makedirs("drift_reports", exist_ok=True)
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
    """Generate and save drift report with Evidently."""
    try:
        reference_data, new_data = _drift_ready(reference_data, new_data)

        if len(new_data) < MIN_DRIFT_ROWS:
            logging.info(f"Skipped drift report: only {len(new_data)} rows (< {MIN_DRIFT_ROWS}).")
            return None, None

        column_mapping = ColumnMapping()
        column_mapping.numerical_features = NUMERICAL_FEATURES
        column_mapping.categorical_features = CATEGORICAL_FEATURES + BINARY_FEATURES
        
        report = Report(metrics=[
            DataDriftPreset(),
            DatasetSummaryMetric(),
            ColumnSummaryMetric(column_name='CreditScore'),
            ColumnSummaryMetric(column_name='Income'),
            ColumnSummaryMetric(column_name='DTIRatio'),
        ])
        
        report.run(
            reference_data=reference_data, 
            current_data=new_data,
            column_mapping=column_mapping
        )
        
        report_path = f"drift_reports/{report_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
        report.save_html(report_path)
        logging.info(f"Drift report saved: {report_path}")
        return report_path, report
    except Exception as e:
        logging.error(f"Error generating drift report: {str(e)}")
        return None, None

def generate_target_drift_report(reference_data, current_data, target_column, report_name="target_drift_report"):
    """Generate target drift report"""
    try:
        if len(current_data) < MIN_DRIFT_ROWS:
            return None, None
            
        column_mapping = ColumnMapping()
        column_mapping.target = target_column
        column_mapping.prediction = 'prediction'
        column_mapping.numerical_features = NUMERICAL_FEATURES
        column_mapping.categorical_features = CATEGORICAL_FEATURES + BINARY_FEATURES
        
        report = Report(metrics=[TargetDriftPreset()])
        report.run(
            reference_data=reference_data,
            current_data=current_data,
            column_mapping=column_mapping
        )
        
        report_path = f"drift_reports/{report_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
        report.save_html(report_path)
        return report_path, report
    except Exception as e:
        logging.error(f"Error generating target drift report: {str(e)}")
        return None, None

def visualize_drift_metrics(reference_data, current_data, artifacts):
    """Create visualizations for data drift analysis"""
    try:
        if len(current_data) < 5:
            return None
            
        ref, cur = _drift_ready(reference_data, current_data)
        
        drift_tab1, drift_tab2, drift_tab3, drift_tab4 = st.tabs([
            "Distribution Comparison", "Statistical Tests", "Feature Drift", "Correlation Changes"
        ])
        
        with drift_tab1:
            st.subheader("Feature Distribution Comparison")
            selected_feature = st.selectbox("Select feature to compare", NUMERICAL_FEATURES)
            
            if selected_feature in ref.columns and selected_feature in cur.columns:
                fig = ff.create_distplot(
                    [ref[selected_feature].dropna(), cur[selected_feature].dropna()],
                    ['Reference', 'Current'],
                    show_hist=False,
                    show_rug=False
                )
                fig.update_layout(
                    title=f"Distribution of {selected_feature}",
                    xaxis_title=selected_feature,
                    yaxis_title="Density"
                )
                st.plotly_chart(fig, use_container_width=True)
                
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Reference Mean", f"{ref[selected_feature].mean():.2f}")
                    st.metric("Reference Std", f"{ref[selected_feature].std():.2f}")
                with col2:
                    st.metric("Current Mean", f"{cur[selected_feature].mean():.2f}", 
                             delta=f"{(cur[selected_feature].mean() - ref[selected_feature].mean()):.2f}")
                    st.metric("Current Std", f"{cur[selected_feature].std():.2f}", 
                             delta=f"{(cur[selected_feature].std() - ref[selected_feature].std()):.2f}")
        
        with drift_tab2:
            st.subheader("Statistical Drift Tests")
            
            drift_scores = {}
            for feature in NUMERICAL_FEATURES:
                if feature in ref.columns and feature in cur.columns:
                    from scipy.stats import ks_2samp
                    stat, p_value = ks_2samp(ref[feature].dropna(), cur[feature].dropna())
                    drift_scores[feature] = p_value
            
            drift_df = pd.DataFrame.from_dict(drift_scores, orient='index', columns=['p_value'])
            drift_df['Drift_Detected'] = drift_df['p_value'] < 0.05
            drift_df['-log(p_value)'] = -np.log10(drift_df['p_value'] + 1e-10)
            
            fig = px.bar(drift_df, x=drift_df.index, y='-log(p_value)', 
                         color='Drift_Detected',
                         title="Statistical Drift Detection (-log p-value)")
            st.plotly_chart(fig, use_container_width=True)
            
            st.dataframe(drift_df.sort_values('p_value'))
        
        with drift_tab3:
            st.subheader("Feature-wise Drift Analysis")
            
            features = []
            drift_scores = []
            for feature in NUMERICAL_FEATURES:
                if feature in ref.columns and feature in cur.columns:
                    from scipy.stats import wasserstein_distance
                    distance = wasserstein_distance(ref[feature].dropna(), cur[feature].dropna())
                    features.append(feature)
                    drift_scores.append(distance)
            
            drift_scores = np.array(drift_scores)
            if len(drift_scores) > 0:
                drift_scores = drift_scores / np.max(drift_scores)
                
                fig = px.bar(x=features, y=drift_scores, 
                            title="Normalized Feature Drift Scores (Wasserstein Distance)")
                st.plotly_chart(fig, use_container_width=True)
        
        with drift_tab4:
            st.subheader("Correlation Changes")
            
            ref_corr = ref[NUMERICAL_FEATURES].corr()
            cur_corr = cur[NUMERICAL_FEATURES].corr()
            
            corr_diff = cur_corr - ref_corr
            
            fig = px.imshow(corr_diff, 
                           title="Correlation Matrix Difference (Current - Reference)",
                           aspect="auto")
            st.plotly_chart(fig, use_container_width=True)
            
        return True
    except Exception as e:
        st.error(f"Error creating drift visualizations: {str(e)}")
        return False

def generate_drift_summary(report):
    """Generate a summary of drift findings"""
    try:
        summary = []
        
        summary.append("## 📊 Data Drift Analysis Summary")
        summary.append(f"**Report Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        summary.append("")
        
        summary.append("### Key Findings")
        
        dataset_drift = False
        drifted_features = []
        
        if hasattr(report, '_inner_suite'):
            metrics = report._inner_suite.workspace[0][0].metrics
            for metric in metrics:
                if hasattr(metric, 'dataset_drift') and metric.dataset_drift:
                    dataset_drift = True
                if hasattr(metric, 'drift_by_columns'):
                    for col, drift_info in metric.drift_by_columns.items():
                        if hasattr(drift_info, 'drift_detected') and drift_info.drift_detected:
                            drifted_features.append(col)
        
        summary.append(f"- **Dataset Drift Detected:** {'Yes' if dataset_drift else 'No'}")
        summary.append(f"- **Number of Drifted Features:** {len(drifted_features)}")
        if drifted_features:
            summary.append(f"- **Drifted Features:** {', '.join(drifted_features[:5])}{'...' if len(drifted_features) > 5 else ''}")
        
        summary.append("")
        summary.append("### Recommendations")
        if dataset_drift or len(drifted_features) > 0:
            summary.append("- ⚠️ **Model Retraining Recommended**: Significant data drift detected")
            summary.append("- 🔍 **Monitor Closely**: Keep tracking these features for changes")
            summary.append("- 📈 **Update Features**: Consider feature engineering or collection changes")
        else:
            summary.append("- ✅ **Model Stability**: No significant data drift detected")
            summary.append("- 📊 **Continue Monitoring**: Regular checks are still recommended")
        
        return "\n".join(summary)
    except Exception as e:
        return f"## 📊 Data Drift Analysis\n\nError generating summary: {str(e)}"

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
    
    if os.path.exists(TARGET_DATA_PATH):
        artifacts['target_data'] = pd.read_csv(TARGET_DATA_PATH)
    
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
        names = []
        names += [f"num__{c}" for c in NUMERICAL_FEATURES]
        names += ["cat__"] * len(CATEGORICAL_FEATURES)
        names += [f"bin__{c}" for c in BINARY_FEATURES]
        return names

def plot_model_feature_importance(artifacts, top_n=20):
    model = artifacts['model']
    pre = artifacts['preprocessor']
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
        feat_names = get_transformed_feature_names(pre)
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
    row = input_df.iloc[0].copy()
    best_row = row.copy()
    best_prob = float(predict_default_probability(pd.DataFrame([row]), artifacts)[0])

    directions = {
        "CreditScore": +1,
        "Income": +1,
        "MonthsEmployed": +1,
        "LoanAmount": -1,
        "InterestRate": -1,
        "DTIRatio": -1,
        "NumCreditLines": -1,
        "Age": 0,
        "LoanTerm": 0,
    }
    
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
    modified = base_df.copy()
    for k, v in overrides.items():
        if k in modified.columns:
            modified.loc[modified.index[0], k] = v
    prob = float(predict_default_probability(modified, artifacts)[0])
    return prob, modified

def make_risk_heatmap(artifacts, base_row: pd.Series,
                      x_feat="CreditScore", y_feat="DTIRatio",
                      x_range=(300, 850), y_range=(0.0, 1.5), x_steps=40, y_steps=40):
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
    fig.add_trace(go.Scatter(
        x=[base_row[x_feat]], y=[base_row[y_feat]],
        mode="markers", marker=dict(size=10, line=dict(width=1, color="black")),
        name="Applicant"
    ))
    return fig

def make_radar_chart(artifacts, input_row: pd.Series, threshold=THRESHOLD):
    ref = artifacts['reference_data'].copy()
    probs_ref = predict_default_probability(ref.copy(), artifacts)
    if probs_ref is None:
        fig = go.Figure()
        fig.add_annotation(text="Unable to compute radar chart.", showarrow=False, x=0.5, y=0.5)
        return fig
    ref = ref.assign(prob=probs_ref)
    def_cohort = ref[ref['prob'] >= threshold]
    nondef_cohort = ref[ref['prob'] < threshold]

    cats = CATEGORICAL_FEATURES + BINARY_FEATURES
    num = NUMERICAL_FEATURES
    mins = ref[num].min()
    maxs = ref[num].max().replace(0, 1)
    def_mean = def_cohort[num].mean()
    nondef_mean = nondef_cohort[num].mean()

    def norm(s):
        return (s - mins) / (maxs - mins + 1e-9)

    applicant_norm = norm(input_row[num])
    def_norm = norm(def_mean)
    nondef_norm = norm(nondef_mean)

    categories = num + [num[0]]
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

# ===== Risk Explanation Engine =====
def generate_risk_explanation(data, prob, artifacts):
    explanations = []
    
    if prob >= THRESHOLD:
        explanations.append("## 🔍 Risk Factor Analysis")
        explanations.append("Your application has been flagged as **High Risk** due to the following factors:")
        
        if data['CreditScore'] < 650:
            explanations.append(f"- **Low Credit Score**: Your credit score of {data['CreditScore']} is below the recommended threshold of 650. This suggests a higher likelihood of payment difficulties based on historical data.")
        
        if data['DTIRatio'] > 0.4:
            explanations.append(f"- **High Debt-to-Income Ratio**: Your DTI ratio of {data['DTIRatio']:.2f} exceeds the recommended maximum of 0.40. This indicates you may have difficulty managing additional debt payments.")
        
        if data['Income'] < 30000 and data['LoanAmount'] > 10000:
            explanations.append(f"- **Income-to-Loan Mismatch**: With an annual income of ${data['Income']:,.0f}, the requested loan amount of ${data['LoanAmount']:,.0f} represents a significant financial burden.")
        
        if data['MonthsEmployed'] < 12:
            explanations.append(f"- **Short Employment History**: You've been employed for only {data['MonthsEmployed']} months. Longer employment history typically indicates more stable income.")
        
        if data['InterestRate'] > 10:
            explanations.append(f"- **High Interest Rate**: The interest rate of {data['InterestRate']:.1f}% suggests the lender perceives higher risk in your application.")
        
        if data['EmploymentType'] in ['Part-time', 'Unemployed']:
            explanations.append(f"- **Employment Status**: Your employment type ({data['EmploymentType']}) may indicate less stable income compared to full-time employment.")
        
        if data['HasDependents'] == 'Yes' and data['Income'] < 50000:
            explanations.append("- **Dependents with Limited Income**: Having dependents while earning a moderate income increases financial pressure and default risk.")
            
        explanations.append("\n## 📊 Model Insights")
        explanations.append("Our machine learning model has analyzed hundreds of similar applications and identified patterns that correlate with higher default rates in your demographic and financial profile.")
        
    else:
        explanations.append("## ✅ Application Strengths")
        explanations.append("Your application shows several positive indicators:")
        
        if data['CreditScore'] >= 700:
            explanations.append(f"- **Strong Credit History**: Your credit score of {data['CreditScore']} demonstrates responsible credit management.")
        
        if data['DTIRatio'] <= 0.35:
            explanations.append(f"- **Healthy Debt-to-Income Ratio**: Your DTI ratio of {data['DTIRatio']:.2f} indicates good debt management capacity.")
        
        if data['MonthsEmployed'] >= 24:
            explanations.append(f"- **Stable Employment**: {data['MonthsEmployed']} months of employment shows income stability.")
            
        if data['HasCoSigner'] == 'Yes':
            explanations.append("- **Co-Signer Available**: Having a co-signer reduces the lender's risk.")
    
    explanations.append("\n## 💡 Recommendations")
    if prob >= THRESHOLD:
        explanations.append("To improve your application, consider:")
        explanations.append("- Increasing your credit score by paying down existing debts")
        explanations.append("- Reducing your debt-to-income ratio")
        explanations.append("- Providing a larger down payment or collateral")
        explanations.append("- Adding a credit-worthy co-signer")
        explanations.append("- Exploring a smaller loan amount")
    else:
        explanations.append("To maintain your strong financial position:")
        explanations.append("- Continue making timely payments on all obligations")
        explanations.append("- Monitor your credit report regularly")
        explanations.append("- Avoid taking on unnecessary new debt")
        explanations.append("- Build emergency savings to cover unexpected expenses")
    
    return "\n\n".join(explanations)

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
        income = col2.number_input("Annual Income ($)", 1000, 500000000, 60000000)
        loan_amount = col3.number_input("Loan Amount ($)", 500, 200000000, 200000000)

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
    st.subheader("Multi-turn Chatbot Assistant")
    
    if 'chatbot_data' not in st.session_state:
        st.session_state.chatbot_data = {}
        st.session_state.chatbot_step = 0
        st.session_state.chatbot_messages = [
            {"role": "assistant", "content": "Hello! I'm here to help you with your loan application. Let's start with some basic information."}
        ]
    
    conversation_steps = [
        {"question": "What is your age?", "field": "Age", "type": "number", "min": 18, "max": 100},
        {"question": "What is your annual income per annum?", "field": "Income", "type": "number", "min": 1000000, "max": 50000000},
        {"question": "How much loan are you requesting?", "field": "LoanAmount", "type": "number", "min": 50000000, "max": 200000000},
        {"question": "What is your credit score?", "field": "CreditScore", "type": "number", "min": 300, "max": 850},
        {"question": "How many months have you been employed at your current job?", "field": "MonthsEmployed", "type": "number", "min": 0, "max": 600},
        {"question": "How many credit lines do you currently have?", "field": "NumCreditLines", "type": "number", "min": 0, "max": 50},
        {"question": "What interest rate are you being offered?", "field": "InterestRate", "type": "number", "min": 0.0, "max": 50.0, "step": 0.1},
        {"question": "What is the loan term in months?", "field": "LoanTerm", "type": "number", "min": 1, "max": 480},
        {"question": "What is your debt-to-income ratio?", "field": "DTIRatio", "type": "number", "min": 0.0, "max": 2.0, "step": 0.01},
        {"question": "What is your highest education level?", "field": "Education", "type": "select", "options": ["High School", "Bachelor's", "Master's", "PhD"]},
        {"question": "What is your employment type?", "field": "EmploymentType", "type": "select", "options": ["Full-time", "Part-time", "Self-employed", "Unemployed"]},
        {"question": "What is the purpose of this loan?", "field": "LoanPurpose", "type": "select", "options": ["Business", "Home", "Education", "Auto"]},
        {"question": "What is your marital status?", "field": "MaritalStatus", "type": "select", "options": ["Single", "Married"]},
        {"question": "Do you have a mortgage?", "field": "HasMortgage", "type": "select", "options": ["No", "Yes"]},
        {"question": "Do you have any dependents?", "field": "HasDependents", "type": "select", "options": ["No", "Yes"]},
        {"question": "Do you have a co-signer?", "field": "HasCoSigner", "type": "select", "options": ["No", "Yes"]},
    ]
    
    for message in st.session_state.chatbot_messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    if st.session_state.chatbot_step < len(conversation_steps):
        current_step = conversation_steps[st.session_state.chatbot_step]
        
        with st.chat_message("assistant"):
            st.markdown(current_step["question"])
        
        if current_step["type"] == "number":
            value = st.number_input(
                current_step["question"],
                min_value=current_step.get("min", 0),
                max_value=current_step.get("max", 1000000),
                step=current_step.get("step", 1),
                key=f"chatbot_{current_step['field']}",
                label_visibility="collapsed"
            )
        else:
            value = st.selectbox(
                current_step["question"],
                options=current_step["options"],
                key=f"chatbot_{current_step['field']}",
                label_visibility="collapsed"
            )
        
        if st.button("Next", key=f"next_{current_step['field']}"):
            st.session_state.chatbot_data[current_step["field"]] = value
            st.session_state.chatbot_messages.append({"role": "user", "content": f"{current_step['field']}: {value}"})
            st.session_state.chatbot_step += 1
            st.rerun()
    
    else:
        df_input = pd.DataFrame([st.session_state.chatbot_data])
        probs = predict_default_probability(df_input, artifacts)
        
        if probs is not None:
            p = float(probs[0])
            prediction = "High Risk" if p >= THRESHOLD else "Low Risk"
            
            with st.chat_message("assistant"):
                st.markdown("## 📊 Loan Application Analysis Complete")
                st.metric("Default Probability", f"{p:.2%}")
                st.metric("Risk Classification", prediction)
                
                risk_explanation = generate_risk_explanation(st.session_state.chatbot_data, p, artifacts)
                st.markdown(risk_explanation)
                
                st.plotly_chart(plot_gauge_simple(p), use_container_width=True)
                
                log_prediction(df_input, {"class": prediction, "prob": p})
        
        if st.button("Start New Application"):
            st.session_state.chatbot_data = {}
            st.session_state.chatbot_step = 0
            st.session_state.chatbot_messages = [
                {"role": "assistant", "content": "Hello! I'm here to help you with your loan application. Let's start with some basic information."}
            ]
            st.rerun()

# ===== Main App =====
def main():
    st.set_page_config(page_title="Loan Default Predictor", page_icon="💰", layout="wide")
    artifacts = load_artifacts()

    st.title("Loan Default Risk Assessment with Advanced Monitoring")
    st.caption(f"Developer: **{DEVELOPER_NAME}** | Model v{MODEL_VERSION} (trained {MODEL_TRAIN_DATE})")

    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "Single Application", "Chatbot Interface", "Batch Processing", 
        "Prediction Logs", "Explainability", "Drift Monitoring"
    ])

    # ---- Single Application ----
    with tab1:
        input_df = get_user_input()
        if input_df is not None:
            probs = predict_default_probability(input_df.copy(), artifacts)
            if probs is not None:
                p = float(probs[0])
                prediction = "High Risk" if p >= THRESHOLD else "Low Risk"

                c1, c2, c3 = st.columns(3)
                c1.metric("Default Probability", f"{p:.2%}")
                c2.metric("Risk Classification", prediction)
                c3.metric("Threshold", f"{THRESHOLD:.0%}")

                st.plotly_chart(gauge_probability(p), use_container_width=True)

                log_prediction(input_df, {"class": prediction, "prob": p})

                st.markdown("## 🔍 Risk Factor Analysis")
                risk_explanation = generate_risk_explanation(input_df.iloc[0].to_dict(), p, artifacts)
                st.markdown(risk_explanation)

                st.markdown("### 🎨 Visual Storytelling")
                base_row = input_df.iloc[0]
                hm = make_risk_heatmap(artifacts, base_row, "CreditScore", "DTIRatio",
                                       x_range=(300, 850), y_range=(0.0, 1.5))
                st.plotly_chart(hm, use_container_width=True)
                radar = make_radar_chart(artifacts, base_row)
                st.plotly_chart(radar, use_container_width=True)

                st.markdown("### 💡 Counterfactual Suggestions")
                suggestions, best_df = suggest_counterfactuals(input_df.copy(), artifacts, THRESHOLD)
                if len(suggestions) == 0:
                    st.info("No actionable single-step changes found within the search budget.")
                else:
                    s_df = pd.DataFrame(suggestions)
                    s_df['new_prob'] = s_df['new_prob'].map(lambda x: f"{x:.2%}")
                    st.dataframe(s_df, use_container_width=True)

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

                report_path, report = generate_drift_report(artifacts['reference_data'], input_df, "single_app")
                if report_path:
                    st.success("Data drift report generated.")
                    with open(report_path, "rb") as f:
                        st.download_button("Download Drift Report (HTML)", f,
                                           file_name=os.path.basename(report_path), key="single_html")
                    with open(report_path, "r", encoding="utf-8") as f:
                        html = f.read()
                    components.html(html, height=600, scrolling=True)
                else:
                    st.info(f"Drift check skipped (need at least {MIN_DRIFT_ROWS} rows).")

                st.markdown("### 📄 Recommendations & Downloadable Report")
                recs = recommendations_for_prob(p)
                st.write("- " + "\n- ".join(recs))
                if st.button("Generate Decision Report (HTML)"):
                    save_to = f"logs/decision_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
                    path = generate_html_report(input_df.copy(), p, prediction, recs, suggestions, save_path=save_to)
                    with open(path, "rb") as f:
                        st.download_button("Download Report (HTML)", f, file_name=os.path.basename(path), mime="text/html")

    # ---- Chatbot Interface ----
    with tab2:
        chatbot_interface(artifacts)

    # ---- Batch Processing ----
    with tab3:
        uploaded_file = st.file_uploader("Upload CSV", type="csv")
        if uploaded_file is not None:
            results_df = process_batch_data(uploaded_file, artifacts)
            if results_df is not None:
                st.success(f"Processed {len(results_df)} applications!")
                high = int((results_df['Risk_Classification'] == "High Risk").sum())
                low = int((results_df['Risk_Classification'] == "Low Risk").sum())
                avg_p = float(results_df['Default_Probability'].mean())
                c1, c2, c3 = st.columns(3)
                c1.metric("High Risk", f"{high}")
                c2.metric("Low Risk", f"{low}")
                c3.metric("Avg Default Prob", f"{avg_p:.2%}")

                ch1, ch2 = st.columns(2)
                with ch1:
                    st.plotly_chart(risk_pie(results_df), use_container_width=True)
                with ch2:
                    st.plotly_chart(prob_hist(results_df), use_container_width=True)

                log_prediction(results_df[FEATURES_ORDER], results_df['Risk_Classification'].tolist())

                report_path, report = generate_drift_report(
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

                st.dataframe(results_df.sort_values('Default_Probability', ascending=False))
                st.download_button(
                    "Download Predictions CSV",
                    data=results_df.to_csv(index=False).encode("utf-8"),
                    file_name="loan_predictions.csv",
                    mime="text/csv"
                )

    # ---- Prediction Logs ----
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

    # ---- Explainability ----
    with tab5:
        st.subheader("Global Explainability (No SHAP)")
        st.plotly_chart(plot_model_feature_importance(artifacts), use_container_width=True)
        st.caption("Tip: Choose two influential features and inspect the Risk Heatmap in the Single Application tab.")

    # ---- Drift Monitoring ----
    with tab6:
        st.header("📊 Advanced Data Drift Monitoring")
        
        col1, col2 = st.columns([2, 1])
        with col1:
            st.subheader("Data Drift Analysis")
            st.info("Upload current data to compare against reference training data")
        
        with col2:
            if st.button("🔄 Refresh Analysis", help="Refresh drift analysis with latest data"):
                st.rerun()
        
        uploaded_drift_file = st.file_uploader(
            "Upload current data for drift analysis", 
            type="csv",
            key="drift_uploader"
        )
        
        if uploaded_drift_file:
            current_data = pd.read_csv(uploaded_drift_file, nrows=MAX_CSV_ROWS)
            current_data = _strip_aux_columns(current_data)
            
            st.success(f"Loaded {len(current_data)} records for drift analysis")
            
            report_path, report = generate_drift_report(
                artifacts['reference_data'], 
                current_data, 
                "comprehensive_drift_report"
            )
            
            if report_path:
                drift_summary = generate_drift_summary(report)
                st.markdown(drift_summary)
                
                st.subheader("Interactive Drift Analysis")
                visualize_drift_metrics(artifacts['reference_data'], current_data, artifacts)
                
                with open(report_path, "rb") as f:
                    st.download_button(
                        "📥 Download Comprehensive Drift Report",
                        f,
                        file_name=os.path.basename(report_path),
                        mime="text/html"
                    )
                
                with st.expander("View Detailed Drift Report"):
                    with open(report_path, "r", encoding="utf-8") as f:
                        html = f.read()
                    components.html(html, height=800, scrolling=True)
            
            if 'target_data' in artifacts and len(current_data) >= MIN_DRIFT_ROWS:
                st.subheader("🎯 Target Drift Analysis")
                
                current_with_pred = current_data.copy()
                probs = predict_default_probability(current_data, artifacts)
                if probs is not None:
                    current_with_pred['prediction'] = probs
                    
                    target_report_path, target_report = generate_target_drift_report(
                        artifacts['reference_data'].assign(
                            target=artifacts['target_data'].iloc[:, 0] if 'target_data' in artifacts else 0
                        ),
                        current_with_pred,
                        'target' if 'target_data' in artifacts else None,
                        "target_drift_report"
                    )
                    
                    if target_report_path:
                        with open(target_report_path, "rb") as f:
                            st.download_button(
                                "📥 Download Target Drift Report",
                                f,
                                file_name=os.path.basename(target_report_path),
                                mime="text/html"
                            )
            
            st.subheader("📈 Drift Monitoring Over Time")
            
            drift_reports = [f for f in os.listdir("drift_reports") if f.endswith('.html')]
            if drift_reports:
                drift_reports.sort(reverse=True)
                selected_report = st.selectbox("Select historical report", drift_reports[:5])
                
                if selected_report:
                    report_path = os.path.join("drift_reports", selected_report)
                    with open(report_path, "rb") as f:
                        st.download_button(
                            "📥 Download Historical Report",
                            f,
                            file_name=selected_report,
                            mime="text/html"
                        )
            else:
                st.info("No historical drift reports found. Upload data to generate reports.")
        
        else:
            st.info("👆 Upload a CSV file to begin drift analysis")
            
            st.subheader("Reference Data Statistics")
            ref_stats = artifacts['reference_data'][NUMERICAL_FEATURES].describe()
            st.dataframe(ref_stats.style.format("{:.2f}"))
            
            st.subheader("Reference Data Distributions")
            selected_feature = st.selectbox("Select feature to view distribution", NUMERICAL_FEATURES)
            
            if selected_feature in artifacts['reference_data'].columns:
                fig = px.histogram(artifacts['reference_data'], x=selected_feature, 
                                  title=f"Distribution of {selected_feature} in Reference Data")
                st.plotly_chart(fig, use_container_width=True)

if __name__ == "__main__":
    try:
        main()
    except Exception:
        st.error("Critical error occurred")
        st.code(traceback.format_exc())
        st.stop()

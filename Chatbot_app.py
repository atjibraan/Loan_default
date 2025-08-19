# chatbot_loan_default_multiturn.py
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import traceback
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, OrdinalEncoder
import plotly.graph_objects as go

# ===== Configuration =====
MODEL_PATH = 'loan_default_model.pkl'
PREPROCESSOR_PATH = 'preprocessor.pkl'
DEVELOPER_NAME = "Jibraan Attar"
MODEL_VERSION = "2.0"
MODEL_TRAIN_DATE = "2025-07-29"
MAX_CSV_ROWS = 2000000
MAX_FILE_SIZE_MB = 25

NUMERICAL_FEATURES = [
    'Age', 'Income', 'LoanAmount', 'CreditScore', 'MonthsEmployed',
    'NumCreditLines', 'InterestRate', 'LoanTerm', 'DTIRatio'
]
CATEGORICAL_FEATURES = ['Education', 'EmploymentType', 'LoanPurpose']
BINARY_FEATURES = ['MaritalStatus', 'HasMortgage', 'HasDependents', 'HasCoSigner']

FEATURES_ORDER = [
    'Age', 'Income', 'LoanAmount', 'CreditScore', 'MonthsEmployed',
    'NumCreditLines', 'InterestRate', 'LoanTerm', 'DTIRatio', 'Education',
    'EmploymentType', 'MaritalStatus', 'HasMortgage', 'HasDependents',
    'LoanPurpose', 'HasCoSigner'
]

# ===== Preprocessor Class =====
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

# ===== Load Artifacts =====
@st.cache_resource
def load_artifacts():
    try:
        model = joblib.load(MODEL_PATH)
        preprocessor = joblib.load(PREPROCESSOR_PATH)
        return {'model': model, 'preprocessor': preprocessor}
    except Exception as e:
        st.error(f"Error loading artifacts: {str(e)}")
        with st.expander("Technical Details"):
            st.code(traceback.format_exc())
        st.stop()

# ===== Prediction =====
def predict_default_probability(input_data, artifacts):
    try:
        for col in NUMERICAL_FEATURES:
            input_data[col] = pd.to_numeric(input_data[col], errors='coerce')
        processed_data = artifacts['preprocessor'].transform(input_data)
        probabilities = artifacts['model'].predict_proba(processed_data)
        return probabilities[:, 1]
    except Exception as e:
        st.error(f"Prediction error: {str(e)}")
        with st.expander("Error details"):
            st.code(traceback.format_exc())
        return None

# ===== Interactive Visuals =====
def plot_gauge(probability):
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=float(probability),
        title={'text': "Default Probability"},
        gauge={'axis': {'range': [0, 1]}, 'bar': {'color': "orange"},
               'steps': [
                   {'range': [0, 0.5], 'color': "lightgreen"},
                   {'range': [0.5, 1], 'color': "red"}]}
    ))
    fig.update_layout(height=300)
    return fig

def plot_probability_bar(probability):
    fig = go.Figure(data=[go.Bar(
        x=['Default', 'No Default'],
        y=[float(probability), 1 - float(probability)],
        marker_color=['#ff7f0e', '#1f77b4']
    )])
    fig.update_layout(yaxis=dict(range=[0,1]), title='Probability Comparison')
    return fig

def plot_risk_pie(df):
    risk_counts = df['Risk_Classification'].value_counts()
    fig = go.Figure(data=[go.Pie(
        labels=risk_counts.index,
        values=risk_counts.values,
        hole=0.4,
        marker_colors=['#ff7f0e','#1f77b4']
    )])
    fig.update_layout(title="Risk Distribution")
    return fig

# ===== Multi-turn Chatbot =====
def chatbot_single_applicant(artifacts):
    st.subheader("💬 Chatbot: Step-by-Step Applicant Info")
    if 'answers' not in st.session_state:
        st.session_state['answers'] = {}
        st.session_state['step'] = 0

    steps = [
        ('Age', 'number', 35),
        ('Income', 'number', 60000),
        ('LoanAmount', 'number', 20000),
        ('CreditScore', 'number', 700),
        ('MonthsEmployed', 'number', 36),
        ('NumCreditLines', 'number', 3),
        ('InterestRate', 'number', 7.5),
        ('LoanTerm', 'number', 36),
        ('DTIRatio', 'number', 0.35),
        ('Education', 'select', ["High School","Bachelor's","Master's","PhD"]),
        ('EmploymentType', 'select', ["Full-time","Part-time","Self-employed","Unemployed"]),
        ('LoanPurpose', 'select', ["Business","Home","Education","Auto"]),
        ('MaritalStatus', 'select', ["Single","Married"]),
        ('HasMortgage', 'select', ["No","Yes"]),
        ('HasDependents', 'select', ["No","Yes"]),
        ('HasCoSigner', 'select', ["No","Yes"])
    ]

    if st.session_state['step'] < len(steps):
        feature, ftype, options = steps[st.session_state['step']]
        st.write(f"**Bot:** Please enter your {feature}")
        if ftype == 'number':
            value = st.number_input(feature, value=options)
        else:
            value = st.selectbox(feature, options)
        if st.button("Next"):
            st.session_state['answers'][feature] = value
            st.session_state['step'] += 1
            st.experimental_rerun()
    else:
        st.write("✅ All information collected!")
        df_input = pd.DataFrame([st.session_state['answers']])
        prob_default = predict_default_probability(df_input, artifacts)[0]
        prediction = "High Risk" if prob_default >= 0.5 else "Low Risk"

        st.metric("Probability of Default", f"{prob_default:.2%}")
        st.metric("Risk Classification", prediction)

        st.plotly_chart(plot_gauge(prob_default), use_container_width=True)
        st.plotly_chart(plot_probability_bar(prob_default), use_container_width=True)

        # Risk explanation
        if prob_default >= 0.5:
            st.error("⚠️ HIGH RISK ⚠️: Consider additional documentation or co-signer")
        else:
            st.success("✅ LOW RISK: Standard approval possible")

        # Reset button
        if st.button("Start Over"):
            st.session_state['answers'] = {}
            st.session_state['step'] = 0
            st.experimental_rerun()

# ===== Batch Upload =====
def batch_upload(artifacts):
    st.subheader("Batch Processing")
    uploaded_file = st.file_uploader("Upload CSV file", type="csv")
    if uploaded_file:
        file_size = len(uploaded_file.getvalue())/(1024*1024)
        if file_size>MAX_FILE_SIZE_MB:
            st.error(f"File too large (> {MAX_FILE_SIZE_MB}MB)")
            return
        df = pd.read_csv(uploaded_file, nrows=MAX_CSV_ROWS)
        if all(col in df.columns for col in FEATURES_ORDER):
            probs = predict_default_probability(df[FEATURES_ORDER], artifacts)
            df['Default_Probability'] = probs
            df['Risk_Classification'] = np.where(df['Default_Probability']>=0.5,"High Risk","Low Risk")
            st.success(f"Processed {len(df)} applicants")
            st.dataframe(df)
            st.plotly_chart(plot_risk_pie(df), use_container_width=True)
            st.download_button("Download CSV", df.to_csv(index=False).encode('utf-8'), file_name="loan_predictions.csv")
        else:
            st.error("Uploaded CSV missing required columns")

# ===== Main App =====
def main():
    st.set_page_config(page_title="Loan Default Chatbot", layout="wide")
    st.title("💬 Loan Default Multi-turn Chatbot")

    artifacts = load_artifacts()
    st.sidebar.header("Developer & Model Info")
    st.sidebar.markdown(f"**Developer:** {DEVELOPER_NAME}")
    st.sidebar.markdown(f"**Model Version:** {MODEL_VERSION}")
    st.sidebar.markdown(f"**Trained on:** {MODEL_TRAIN_DATE}")

    tab1, tab2 = st.tabs(["Single Applicant Chatbot", "Batch Upload"])
    with tab1:
        chatbot_single_applicant(artifacts)
    with tab2:
        batch_upload(artifacts)

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        st.error("Critical error occurred!")
        with st.expander("Error details"):
            st.code(traceback.format_exc())

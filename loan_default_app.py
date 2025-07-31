import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.compose import ColumnTransformer

# =============================
# Embedded Preprocessor Class
# =============================
class Preprocessor(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.num_cols = [
            'Age', 'Income', 'LoanAmount', 'CreditScore', 'MonthsEmployed',
            'NumCreditLines', 'InterestRate', 'LoanTerm', 'DTIRatio'
        ]
        self.cat_cols_onehot = ['Education', 'EmploymentType', 'LoanPurpose']
        self.cat_cols_ordinal = ['MaritalStatus']
        self.binary_cols = ['HasMortgage', 'HasDependents', 'HasCoSigner']

        self.ordinal_mapping = [['Single', 'Married', 'Divorced', 'Widowed']]
        self.scaler = StandardScaler()
        self.onehot = OneHotEncoder(handle_unknown='ignore', sparse=False)
        self.ordinal = OrdinalEncoder(categories=self.ordinal_mapping)
        self.pipeline = None

    def fit(self, X, y=None):
        transformers = [
            ('num', self.scaler, self.num_cols),
            ('onehot', self.onehot, self.cat_cols_onehot),
            ('ordinal', self.ordinal, self.cat_cols_ordinal)
        ]
        self.pipeline = ColumnTransformer(transformers, remainder='passthrough')
        self.pipeline.fit(X)
        return self

    def transform(self, X):
        return self.pipeline.transform(X)

# =============================
# Load Model and Setup
# =============================
@st.cache_resource
def load_model():
    model = joblib.load('loan_default_model.pkl')
    return model

model = load_model()
preprocessor = Preprocessor()

# =============================
# UI Layout
# =============================
st.set_page_config(page_title="Loan Default Predictor", layout="centered")
st.title("💸 Loan Default Prediction App")

st.markdown("""
This app predicts whether a loan will default based on applicant and loan details.
- Model: `XGBoost`
- Preprocessing: Embedded (no separate file needed)
""")

# =============================
# User Input
# =============================
st.subheader("🔍 Enter Applicant Details")

with st.expander("🧍 Predict One Applicant"):
    def user_input():
        data = {
            'Age': st.number_input('Age', 18, 100, 30),
            'Income': st.number_input('Income', 1000, 1000000, 50000),
            'LoanAmount': st.number_input('Loan Amount', 500, 500000, 10000),
            'CreditScore': st.slider('Credit Score', 300, 850, 700),
            'MonthsEmployed': st.number_input('Months Employed', 0, 600, 24),
            'NumCreditLines': st.number_input('Number of Credit Lines', 1, 20, 5),
            'InterestRate': st.slider('Interest Rate (%)', 1.0, 50.0, 10.0),
            'LoanTerm': st.number_input('Loan Term (months)', 6, 360, 60),
            'DTIRatio': st.slider('Debt-to-Income Ratio', 0.0, 2.0, 0.3),
            'Education': st.selectbox('Education', ['High School', 'Bachelors', 'Masters', 'PhD']),
            'EmploymentType': st.selectbox('Employment Type', ['Salaried', 'Self-Employed', 'Unemployed']),
            'MaritalStatus': st.selectbox('Marital Status', ['Single', 'Married', 'Divorced', 'Widowed']),
            'HasMortgage': st.radio('Has Mortgage?', ['Yes', 'No']) == 'Yes',
            'HasDependents': st.radio('Has Dependents?', ['Yes', 'No']) == 'Yes',
            'LoanPurpose': st.selectbox('Loan Purpose', ['Home', 'Car', 'Education', 'Business', 'Other']),
            'HasCoSigner': st.radio('Has Co-Signer?', ['Yes', 'No']) == 'Yes'
        }
        return pd.DataFrame([data])
    
    input_df = user_input()

    if st.button("Predict Default (Single Entry)"):
        preprocessor.fit(input_df)
        transformed = preprocessor.transform(input_df)
        prediction = model.predict(transformed)[0]
        st.success(f"Prediction: {'🔴 Default' if prediction else '🟢 No Default'}")

# =============================
# CSV Upload Section
# =============================
st.subheader("📂 Predict in Bulk with CSV Upload")

with st.expander("📁 Upload CSV for Batch Prediction"):
    uploaded_file = st.file_uploader("Upload CSV with 16 columns", type=['csv'])

    if uploaded_file:
        batch_df = pd.read_csv(uploaded_file)
        try:
            preprocessor.fit(batch_df)
            transformed_batch = preprocessor.transform(batch_df)
            predictions = model.predict(transformed_batch)
            batch_df['Prediction'] = np.where(predictions == 1, 'Default', 'No Default')
            st.write(batch_df)

            fig, ax = plt.subplots()
            sns.countplot(x='Prediction', data=batch_df, ax=ax, palette='pastel')
            st.pyplot(fig)
        except Exception as e:
            st.error(f"Prediction failed: {e}")

# =============================
# Legal Disclaimer
# =============================
st.caption("⚠️ This tool is for demonstration purposes only. Do not use for actual financial decisions.")




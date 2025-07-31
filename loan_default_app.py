
import streamlit as st
import pandas as pd
import numpy as np
import joblib
from datetime import datetime
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.base import BaseEstimator, TransformerMixin

# Re-create the custom Preprocessor class (as used in training)
class Preprocessor(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.numerical_features = [
            'Age', 'Income', 'LoanAmount', 'CreditScore', 'MonthsEmployed',
            'NumCreditLines', 'InterestRate', 'LoanTerm', 'DTIRatio'
        ]
        self.categorical_features = [
            'Education', 'EmploymentType', 'LoanPurpose'
        ]
        self.binary_features = [
            'MaritalStatus', 'HasMortgage', 'HasDependents', 'HasCoSigner'
        ]
        self.column_transformer = ColumnTransformer(
            transformers=[
                ('num', StandardScaler(), self.numerical_features),
                ('cat', OneHotEncoder(drop='first', sparse_output=False), self.categorical_features),
                ('bin', OrdinalEncoder(), self.binary_features)
            ]
        )

    def fit(self, X, y=None):
        self.column_transformer.fit(X)
        return self

    def transform(self, X):
        return self.column_transformer.transform(X)

# Load pre-trained artifacts
@st.cache_resource(ttl=86400)
def load_artifacts():
    preprocessor = joblib.load("preprocessor.pkl")
    model = joblib.load("loan_default_model.pkl")
    return preprocessor, model

preprocessor, model = load_artifacts()

# Streamlit App Configuration
st.set_page_config(page_title="Loan Default Predictor", layout="wide")
st.title("📊 Loan Default Prediction App")

# Define Input Form
st.subheader("Single Applicant Prediction")
with st.form("prediction_form"):
    col1, col2 = st.columns(2)
    with col1:
        age = st.number_input("Age", 18, 100, 30)
        income = st.number_input("Annual Income", 1000, 1_000_000, 50000)
        loan_amount = st.number_input("Loan Amount", 1000, 1_000_000, 10000)
        credit_score = st.number_input("Credit Score", 300, 850, 600)
        months_employed = st.number_input("Months Employed", 0, 480, 60)
        num_credit_lines = st.number_input("Number of Credit Lines", 0, 20, 3)
        interest_rate = st.number_input("Interest Rate (%)", 1.0, 100.0, 10.0)
    with col2:
        loan_term = st.number_input("Loan Term (months)", 6, 360, 60)
        dti_ratio = st.number_input("DTI Ratio", 0.0, 5.0, 1.0)
        education = st.selectbox("Education", ['High School', "Bachelor's", "Master's", 'PhD'])
        employment_type = st.selectbox("Employment Type", ['Full-time', 'Part-time', 'Unemployed', 'Self-employed'])
        loan_purpose = st.selectbox("Loan Purpose", ['Other', 'Auto', 'Business', 'Home', 'Education'])
        marital_status = st.selectbox("Marital Status", ['Single', 'Married'])
        has_mortgage = st.selectbox("Has Mortgage", ['Yes', 'No'])
        has_dependents = st.selectbox("Has Dependents", ['Yes', 'No'])
        has_cosigner = st.selectbox("Has Co-Signer", ['Yes', 'No'])

    submitted = st.form_submit_button("🔮 Predict")

if submitted:
    try:
        input_data = pd.DataFrame([{
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
            'MaritalStatus': marital_status,
            'HasMortgage': has_mortgage,
            'HasDependents': has_dependents,
            'LoanPurpose': loan_purpose,
            'HasCoSigner': has_cosigner
        }])

        processed = preprocessor.transform(input_data)
        prediction = model.predict(processed)[0]
        probability = model.predict_proba(processed)[0][1]

        st.subheader("Prediction Result")
        if prediction == 1:
            st.error(f"🚨 High Risk of Default: {probability:.2%} probability")
        else:
            st.success(f"✅ Low Risk of Default: {probability:.2%} probability")
    except Exception as e:
        st.error(f"Prediction failed: {e}")

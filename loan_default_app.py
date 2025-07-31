import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
import matplotlib.pyplot as plt
import seaborn as sns

# Custom Preprocessor class (must match training)
class Preprocessor(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.numerical_features = [
            'Age', 'Income', 'LoanAmount', 'CreditScore', 'MonthsEmployed',
            'NumCreditLines', 'InterestRate', 'LoanTerm', 'DTIRatio'
        ]
        self.categorical_features = ['Education', 'EmploymentType', 'LoanPurpose']
        self.binary_features = ['MaritalStatus', 'HasMortgage', 'HasDependents', 'HasCoSigner']
        self.column_transformer = ColumnTransformer(transformers=[
            ('num', StandardScaler(), self.numerical_features),
            ('cat', OneHotEncoder(drop='first', sparse_output=False), self.categorical_features),
            ('bin', OrdinalEncoder(), self.binary_features)
        ])

    def fit(self, X, y=None):
        self.column_transformer.fit(X)
        return self

    def transform(self, X):
        return self.column_transformer.transform(X)

# Load model and preprocessor
@st.cache_resource(ttl=86400)
def load_artifacts():
    preprocessor = joblib.load("preprocessor.pkl")
    model = joblib.load("loan_default_model.pkl")
    return preprocessor, model

preprocessor, model = load_artifacts()

st.set_page_config(page_title="Loan Default Predictor - Jibraan_Predicts", layout="wide")
st.title("📊 Loan Default Predictor")
st.markdown("**by Jibraan_Predicts**")

# Sidebar Info
st.sidebar.header("Model Details")
st.sidebar.markdown("**Model**: XGBoost Classifier")
st.sidebar.markdown("**Features Used**: 16 total")
st.sidebar.markdown("- 9 Numerical\n- 3 Categorical\n- 4 Binary")
st.sidebar.markdown("**Preprocessing**: StandardScaler + OneHotEncoder + OrdinalEncoder")
st.sidebar.markdown("**Imbalance Handling**: SMOTE + RandomUnderSampler")

# --- Single Input Form ---
st.subheader("🔹 Single Applicant Prediction")
with st.form("predict_form"):
    col1, col2 = st.columns(2)
    with col1:
        age = st.number_input("Age", 18, 100, 30)
        income = st.number_input("Annual Income", 1000, 1_000_000, 50000)
        loan_amount = st.number_input("Loan Amount", 1000, 1_000_000, 10000)
        credit_score = st.number_input("Credit Score", 300, 850, 600)
        months_employed = st.number_input("Months Employed", 0, 600, 60)
        num_credit_lines = st.number_input("Number of Credit Lines", 0, 20, 3)
        interest_rate = st.number_input("Interest Rate (%)", 0.0, 100.0, 10.0)
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

    submit = st.form_submit_button("🔮 Predict")

if submit:
    try:
        row = pd.DataFrame([{
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
        }])
        processed = preprocessor.transform(row)
        pred = model.predict(processed)[0]
        prob = model.predict_proba(processed)[0][1]
        st.markdown("## 🎯 Prediction Result")
        if pred == 1:
            st.error(f"🚨 High Default Risk: {prob:.2%}")
        else:
            st.success(f"✅ Low Default Risk: {prob:.2%}")

        # Feature importance-like explanation
        st.markdown("### 🔍 Why this prediction?")
        fig, ax = plt.subplots()
        feature_values = row.iloc[0].values
        labels = row.columns
        sns.barplot(x=feature_values, y=labels, ax=ax, palette="coolwarm")
        ax.set_title("Input Feature Values")
        st.pyplot(fig)
    except Exception as e:
        st.error(f"Prediction failed: {e}")

# --- CSV Upload Section ---
st.subheader("📁 Batch Prediction via CSV Upload")
csv_file = st.file_uploader("Upload CSV with 16 input columns", type=["csv"])
if csv_file:
    try:
        df = pd.read_csv(csv_file)
        processed = preprocessor.transform(df)
        predictions = model.predict(processed)
        probabilities = model.predict_proba(processed)[:, 1]

        df['Prediction'] = predictions
        df['Probability'] = probabilities
        st.dataframe(df)

        # Show counts
        st.markdown("### 📈 Prediction Summary")
        fig, ax = plt.subplots()
        sns.countplot(x='Prediction', data=df, palette='Set2', ax=ax)
        ax.set_xticklabels(['No Default', 'Default'])
        st.pyplot(fig)
    except Exception as e:
        st.error(f"Batch prediction failed: {e}")


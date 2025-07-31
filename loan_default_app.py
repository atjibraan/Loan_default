import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.combine import SMOTEENN

from xgboost import XGBClassifier

# =============================
# 1. Embedded Preprocessor Class
# =============================
class Preprocessor:
    def __init__(self):
        self.numeric_features = ['Age', 'Income', 'LoanAmount', 'CreditScore', 'MonthsEmployed',
                                 'NumCreditLines', 'InterestRate', 'LoanTerm', 'DTIRatio']
        self.ord_features = ['Education']
        self.cat_features = ['EmploymentType', 'MaritalStatus', 'HasMortgage',
                             'HasDependents', 'LoanPurpose', 'HasCoSigner']

        self.ordinal_encoder = OrdinalEncoder()
        self.onehot_encoder = OneHotEncoder(handle_unknown='ignore', sparse=False)
        self.scaler = StandardScaler()

        self.column_transformer = ColumnTransformer(transformers=[
            ('num', self.scaler, self.numeric_features),
            ('ord', self.ordinal_encoder, self.ord_features),
            ('cat', self.onehot_encoder, self.cat_features)
        ])

    def fit(self, X, y=None):
        self.column_transformer.fit(X)
        return self

    def transform(self, X):
        return self.column_transformer.transform(X)

    def fit_transform(self, X, y=None):
        return self.column_transformer.fit_transform(X)

# =============================
# 2. Load Trained Model
# =============================
@st.cache_resource
def load_model():
    return joblib.load("loan_default_model.pkl")

model = load_model()

# =============================
# 3. Streamlit App
# =============================
st.set_page_config(page_title="Loan Default Prediction App", layout="centered")
st.title("💰 Loan Default Prediction")
st.markdown("Upload a borrower's information to predict whether they are likely to default on their loan.")

# =============================
# 4. Input Form or File Upload
# =============================
st.sidebar.header("📤 Upload CSV File or Use Form")

input_mode = st.sidebar.radio("Choose Input Mode", ["Manual Entry", "Upload CSV"])

# Input features
feature_names = ['Age', 'Income', 'LoanAmount', 'CreditScore', 'MonthsEmployed',
                 'NumCreditLines', 'InterestRate', 'LoanTerm', 'DTIRatio',
                 'Education', 'EmploymentType', 'MaritalStatus', 'HasMortgage',
                 'HasDependents', 'LoanPurpose', 'HasCoSigner']

preprocessor = Preprocessor()

def make_prediction(df_input):
    X = preprocessor.fit_transform(df_input)
    pred = model.predict(X)
    proba = model.predict_proba(X)[:, 1]
    return pred, proba

if input_mode == "Manual Entry":
    with st.form("prediction_form"):
        col1, col2 = st.columns(2)

        with col1:
            age = st.number_input("Age", 18, 100, 30)
            income = st.number_input("Income", 0, 1000000, 50000)
            loan_amount = st.number_input("Loan Amount", 0, 1000000, 20000)
            credit_score = st.number_input("Credit Score", 300, 850, 650)
            months_employed = st.number_input("Months Employed", 0, 480, 60)
            num_credit_lines = st.number_input("Number of Credit Lines", 0, 50, 5)
            interest_rate = st.number_input("Interest Rate (%)", 0.0, 100.0, 7.5)

        with col2:
            loan_term = st.number_input("Loan Term (months)", 6, 360, 60)
            dti_ratio = st.number_input("DTI Ratio", 0.0, 5.0, 0.3)
            education = st.selectbox("Education", ["High School", "Bachelor", "Master", "PhD"])
            employment_type = st.selectbox("Employment Type", ["Salaried", "Self-Employed", "Unemployed"])
            marital_status = st.selectbox("Marital Status", ["Single", "Married", "Divorced"])
            has_mortgage = st.selectbox("Has Mortgage", ["Yes", "No"])
            has_dependents = st.selectbox("Has Dependents", ["Yes", "No"])
            loan_purpose = st.selectbox("Loan Purpose", ["Personal", "Business", "Education", "Medical"])
            has_cosigner = st.selectbox("Has Co-Signer", ["Yes", "No"])

        submitted = st.form_submit_button("Predict")

        if submitted:
            input_dict = {
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
            }

            df_input = pd.DataFrame([input_dict])
            pred, proba = make_prediction(df_input)

            st.subheader("📊 Prediction Result")
            st.success(f"Prediction: {'Default' if pred[0]==1 else 'No Default'}")
            st.info(f"Probability of Default: {proba[0]*100:.2f}%")

else:
    uploaded_file = st.file_uploader("Upload a CSV file with all 16 features", type=["csv"])

    if uploaded_file:
        try:
            df_uploaded = pd.read_csv(uploaded_file)

            missing_cols = set(feature_names) - set(df_uploaded.columns)
            if missing_cols:
                st.error(f"Missing required columns: {missing_cols}")
            else:
                pred, proba = make_prediction(df_uploaded)

                df_uploaded['Prediction'] = ["Default" if p == 1 else "No Default" for p in pred]
                df_uploaded['Default Probability (%)'] = (proba * 100).round(2)

                st.subheader("📈 Batch Prediction Results")
                st.dataframe(df_uploaded)

                st.download_button("Download Results", df_uploaded.to_csv(index=False), "predictions.csv", "text/csv")
        except Exception as e:
            st.error(f"Error processing the file: {e}")

# =============================
# 5. About / Metadata
# =============================
st.markdown("---")
with st.expander("📘 About this App"):
    st.markdown("""
    - ✅ **Model**: XGBoost trained on 16 financial and demographic features.
    - ✅ **Preprocessing**: StandardScaler + Ordinal + OneHot Encoding
    - ✅ **Balancing**: SMOTE + RandomUnderSampler
    - ✅ **Supports**: Manual form and CSV batch prediction
    - ⚠️ **Disclaimer**: This tool is for demonstration purposes only. Not financial advice.
    """)






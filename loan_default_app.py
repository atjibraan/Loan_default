import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
import matplotlib.pyplot as plt
import seaborn as sns

# =============================
# Custom Preprocessor Class
# =============================
class Preprocessor(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.numerical_features = [
            'Age', 'Income', 'LoanAmount', 'CreditScore', 'MonthsEmployed',
            'NumCreditLines', 'InterestRate', 'LoanTerm', 'DTIRatio'
        ]
        self.categorical_features = ['Education', 'EmploymentType', 'LoanPurpose']
        self.binary_features = ['MaritalStatus', 'HasMortgage', 'HasDependents', 'HasCoSigner']
        self.column_transformer = ColumnTransformer(
            transformers=[
                ('num', StandardScaler(), self.numerical_features),
                ('cat', OneHotEncoder(drop='first', sparse_output=False), self.categorical_features),
                ('bin', OrdinalEncoder(), self.binary_features)
            ]
        )

    def fit(self, X, y=None):
        return self.column_transformer.fit(X, y)

    def transform(self, X):
        return self.column_transformer.transform(X)

# =============================
# Load Model and Preprocessor
# =============================
@st.cache_resource(ttl=86400)
def load_artifacts():
    model = joblib.load("loan_default_model.pkl")
    preprocessor = joblib.load("preprocessor.pkl")
    return model, preprocessor

model, preprocessor = load_artifacts()

# =============================
# Streamlit App UI
# =============================
st.set_page_config(page_title="Loan Default Predictor - Jibraan Predicts", layout="wide")
st.title("💼 Loan Default Predictor")
st.caption("by **Jibraan_Predicts**")

tab1, tab2 = st.tabs(["🔮 Single Prediction", "📂 Batch Prediction (CSV)"])

# ========== TAB 1 ==========
with tab1:
    with st.form("user_input_form"):
        st.subheader("Enter Applicant Details")
        col1, col2 = st.columns(2)
        with col1:
            Age = st.number_input("Age", 18, 100, 30)
            Income = st.number_input("Annual Income", 1000, 1_000_000, 50000)
            LoanAmount = st.number_input("Loan Amount", 1000, 1_000_000, 10000)
            CreditScore = st.number_input("Credit Score", 300, 850, 600)
            MonthsEmployed = st.number_input("Months Employed", 0, 480, 60)
            NumCreditLines = st.number_input("Number of Credit Lines", 0, 20, 3)
            InterestRate = st.number_input("Interest Rate (%)", 1.0, 100.0, 10.0)
        with col2:
            LoanTerm = st.number_input("Loan Term (months)", 6, 360, 60)
            DTIRatio = st.number_input("DTI Ratio", 0.0, 5.0, 1.0)
            Education = st.selectbox("Education", ["High School", "Bachelor's", "Master's", "PhD"])
            EmploymentType = st.selectbox("Employment Type", ["Full-time", "Part-time", "Unemployed", "Self-employed"])
            LoanPurpose = st.selectbox("Loan Purpose", ["Other", "Auto", "Business", "Home", "Education"])
            MaritalStatus = st.selectbox("Marital Status", ["Single", "Married"])
            HasMortgage = st.selectbox("Has Mortgage", ["Yes", "No"])
            HasDependents = st.selectbox("Has Dependents", ["Yes", "No"])
            HasCoSigner = st.selectbox("Has Co-Signer", ["Yes", "No"])

        submit_btn = st.form_submit_button("Predict")

    if submit_btn:
        try:
            input_df = pd.DataFrame([{
                "Age": Age,
                "Income": Income,
                "LoanAmount": LoanAmount,
                "CreditScore": CreditScore,
                "MonthsEmployed": MonthsEmployed,
                "NumCreditLines": NumCreditLines,
                "InterestRate": InterestRate,
                "LoanTerm": LoanTerm,
                "DTIRatio": DTIRatio,
                "Education": Education,
                "EmploymentType": EmploymentType,
                "LoanPurpose": LoanPurpose,
                "MaritalStatus": MaritalStatus,
                "HasMortgage": HasMortgage,
                "HasDependents": HasDependents,
                "HasCoSigner": HasCoSigner
            }])

            X_processed = preprocessor.transform(input_df)
            prediction = model.predict(X_processed)[0]
            proba = model.predict_proba(X_processed)[0][1]

            st.subheader("📈 Prediction Result")
            if prediction == 1:
                st.error(f"❌ High Risk of Default - Probability: {proba:.2%}")
            else:
                st.success(f"✅ Low Risk of Default - Probability: {proba:.2%}")

            # Visual Explanation
            st.subheader("📊 Risk Factors")
            fig, ax = plt.subplots(figsize=(10, 4))
            input_df.T.plot(kind="barh", legend=False, ax=ax)
            ax.set_title("Applicant Features")
            st.pyplot(fig)

        except Exception as e:
            st.error(f"Prediction Error: {e}")

# ========== TAB 2 ==========
with tab2:
    st.subheader("📂 Upload CSV File for Batch Prediction")
    uploaded_file = st.file_uploader("Upload CSV", type=["csv"])
    if uploaded_file:
        try:
            df = pd.read_csv(uploaded_file)
            X_batch = preprocessor.transform(df)
            preds = model.predict(X_batch)
            probs = model.predict_proba(X_batch)[:, 1]

            df["Prediction"] = preds
            df["Default Probability"] = probs
            st.success("✅ Batch Prediction Completed")
            st.dataframe(df)

            # Show pie chart
            st.subheader("🧮 Default Risk Distribution")
            risk_count = df["Prediction"].value_counts().rename({0: "No Default", 1: "Default"})
            fig2, ax2 = plt.subplots()
            ax2.pie(risk_count, labels=risk_count.index, autopct='%1.1f%%', startangle=90, colors=["#28a745", "#dc3545"])
            ax2.axis("equal")
            st.pyplot(fig2)

            st.download_button("📥 Download Results", df.to_csv(index=False), file_name="loan_predictions.csv")
        except Exception as e:
            st.error(f"Error during batch processing: {e}")



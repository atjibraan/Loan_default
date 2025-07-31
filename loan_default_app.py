
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Load artifacts
@st.cache_resource
def load_model():
    return joblib.load("loan_default_model.pkl")

@st.cache_resource
def load_preprocessor():
    return joblib.load("preprocessor.pkl")

model = load_model()
preprocessor = load_preprocessor()

# Features list
features = [
    'Age', 'Income', 'LoanAmount', 'CreditScore', 'MonthsEmployed',
    'NumCreditLines', 'InterestRate', 'LoanTerm', 'DTIRatio', 'Education',
    'EmploymentType', 'MaritalStatus', 'HasMortgage', 'HasDependents',
    'LoanPurpose', 'HasCoSigner'
]

# Page configuration
st.set_page_config(page_title="Loan Default Prediction App", layout="wide")
st.title("Loan Default Prediction App")
st.markdown("Predict the likelihood of a loan default using a trained ML model.")

# Sidebar
st.sidebar.title("Navigation")
option = st.sidebar.radio("Choose Input Type:", ["Single Application", "Batch Upload"])

# Input form for single application
if option == "Single Application":
    st.header("Single Applicant Prediction")
    with st.form("applicant_form"):
        col1, col2, col3 = st.columns(3)

        with col1:
            Age = st.number_input("Age", 18, 100, 30)
            Income = st.number_input("Annual Income ($)", 0, 1000000, 50000)
            LoanAmount = st.number_input("Loan Amount ($)", 1000, 1000000, 10000)
            CreditScore = st.slider("Credit Score", 300, 850, 650)
            MonthsEmployed = st.number_input("Months Employed", 0, 600, 24)

        with col2:
            NumCreditLines = st.number_input("Number of Credit Lines", 0, 50, 5)
            InterestRate = st.number_input("Interest Rate (%)", 0.0, 50.0, 5.0)
            LoanTerm = st.number_input("Loan Term (months)", 6, 360, 60)
            DTIRatio = st.slider("Debt-to-Income Ratio", 0.0, 1.0, 0.2)
            Education = st.selectbox("Education", ["High School", "Bachelors", "Masters", "PhD"])

        with col3:
            EmploymentType = st.selectbox("Employment Type", ["Salaried", "Self-Employed", "Unemployed"])
            MaritalStatus = st.selectbox("Marital Status", ["Single", "Married", "Divorced"])
            HasMortgage = st.selectbox("Has Mortgage?", ["No", "Yes"])
            HasDependents = st.selectbox("Has Dependents?", ["No", "Yes"])
            LoanPurpose = st.selectbox("Loan Purpose", ["Home", "Car", "Education", "Personal", "Business"])
            HasCoSigner = st.selectbox("Has Co-Signer?", ["No", "Yes"])

        submitted = st.form_submit_button("Predict")

        if submitted:
            input_dict = {
                'Age': Age,
                'Income': Income,
                'LoanAmount': LoanAmount,
                'CreditScore': CreditScore,
                'MonthsEmployed': MonthsEmployed,
                'NumCreditLines': NumCreditLines,
                'InterestRate': InterestRate,
                'LoanTerm': LoanTerm,
                'DTIRatio': DTIRatio,
                'Education': Education,
                'EmploymentType': EmploymentType,
                'MaritalStatus': MaritalStatus,
                'HasMortgage': HasMortgage,
                'HasDependents': HasDependents,
                'LoanPurpose': LoanPurpose,
                'HasCoSigner': HasCoSigner
            }

            input_df = pd.DataFrame([input_dict])
            input_processed = preprocessor.transform(input_df)
            prediction_proba = model.predict_proba(input_processed)[0][1]
            prediction = model.predict(input_processed)[0]

            st.subheader("Prediction Result")
            st.write(f"**Default Probability:** {prediction_proba:.2%}")
            st.write(f"**Prediction:** {'Likely to Default' if prediction == 1 else 'Not Likely to Default'}")

            st.subheader("Why this applicant is at risk?")
            st.markdown("The following chart highlights the most influential features contributing to the predicted default probability.")
            st.image("shap_summary_example.png", caption="Top Features Impacting Default Risk", use_column_width=True)

# Batch prediction from CSV
elif option == "Batch Upload":
    st.header("Batch Prediction from CSV")
    uploaded_file = st.file_uploader("Upload CSV File", type=["csv"])

    if uploaded_file:
        try:
            data = pd.read_csv(uploaded_file)
            if not set(features).issubset(data.columns):
                st.error("Uploaded CSV does not contain the required columns.")
            else:
                processed_data = preprocessor.transform(data[features])
                predictions = model.predict(processed_data)
                probabilities = model.predict_proba(processed_data)[:, 1]

                result_df = data.copy()
                result_df['Default Probability'] = probabilities
                result_df['Prediction'] = np.where(predictions == 1, 'Likely to Default', 'Not Likely to Default')

                st.success("Batch prediction completed.")
                st.dataframe(result_df.head(10))

                csv = result_df.to_csv(index=False).encode('utf-8')
                st.download_button("Download Predictions", csv, "batch_predictions.csv", "text/csv")

        except Exception as e:
            st.error(f"Error processing file: {str(e)}")

# Footer
st.markdown("---")
st.markdown("**Developed by: Your Name Here**")
st.markdown("**Model**: Gradient Boosting Classifier (trained with sklearn)")
st.markdown("**Version**: 1.0 | © 2025")

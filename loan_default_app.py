import streamlit as st
import pandas as pd
import numpy as np
import joblib
from xgboost import XGBClassifier
import matplotlib.pyplot as plt

# Load model and preprocessor
model = joblib.load("loan_default_model.pkl")
preprocessor = joblib.load("preprocessor.pkl")

# Feature lists (must match training script)
numerical_features = [
    'Age', 'Income', 'LoanAmount', 'CreditScore', 'MonthsEmployed',
    'NumCreditLines', 'InterestRate', 'LoanTerm', 'DTIRatio'
]

categorical_features = ['Education', 'EmploymentType', 'LoanPurpose']
binary_features = ['MaritalStatus', 'HasMortgage', 'HasDependents', 'HasCoSigner']

all_features = numerical_features + categorical_features + binary_features

# UI title
st.title("Loan Default Prediction App")
st.markdown("Predict the likelihood of a customer defaulting on a loan. Powered by XGBoost.")

# Tabs for real-time input and CSV upload
tab1, tab2 = st.tabs(["🔍 Real-Time Prediction", "📂 Batch Prediction via CSV"])

with tab1:
    st.subheader("Enter Customer Details")
    input_data = {}

    for col in numerical_features:
        input_data[col] = st.number_input(f"{col}", value=0.0, step=1.0)

    for col in categorical_features:
        input_data[col] = st.selectbox(f"{col}", options=["High School", "Graduate", "Post-Graduate"] if col == "Education"
                                       else ["Salaried", "Self-Employed", "Unemployed"] if col == "EmploymentType"
                                       else ["Personal", "Business", "Education", "Home", "Other"])

    for col in binary_features:
        input_data[col] = st.selectbox(f"{col}", options=["No", "Yes"])

    if st.button("Predict Default"):
        input_df = pd.DataFrame([input_data])
        preprocessed = preprocessor.transform(input_df)
        probability = model.predict_proba(preprocessed)[0, 1]
        prediction = model.predict(preprocessed)[0]

        st.markdown(f"### 📊 Prediction: {'Default' if prediction == 1 else 'No Default'}")
        st.markdown(f"### 💡 Probability of Default: {probability:.2%}")

with tab2:
    st.subheader("Upload CSV File")
    uploaded_file = st.file_uploader("Upload a CSV file with customer data", type=["csv"])

    if uploaded_file:
        try:
            df = pd.read_csv(uploaded_file)

            missing_cols = [col for col in all_features if col not in df.columns]
            if missing_cols:
                st.error(f"Missing required columns: {missing_cols}")
            else:
                processed = preprocessor.transform(df)
                preds = model.predict(processed)
                probs = model.predict_proba(processed)[:, 1]

                df['Prediction'] = np.where(preds == 1, 'Default', 'No Default')
                df['Probability'] = probs

                st.success("Predictions completed.")
                st.dataframe(df[['Prediction', 'Probability'] + all_features])

                # Download option
                csv_out = df.to_csv(index=False).encode('utf-8')
                st.download_button("Download Predictions", data=csv_out, file_name="loan_predictions.csv", mime="text/csv")

        except Exception as e:
            st.error(f"Error processing file: {str(e)}")

# Footer
st.markdown("---")
st.caption("Developed using Streamlit and XGBoost | © 2025")




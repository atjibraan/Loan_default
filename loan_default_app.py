import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

# Set page config
st.set_page_config(page_title="Loan Default Prediction", layout="wide")

# Load model and preprocessor
@st.cache_resource
def load_model():
    model = joblib.load("loan_default_model.pkl")
    preprocessor = joblib.load("preprocessor.pkl")
    return model, preprocessor

model, preprocessor = load_model()

# Feature names in the order expected by the model
FEATURES = [
    'Age', 'Income', 'LoanAmount', 'CreditScore', 'MonthsEmployed',
    'NumCreditLines', 'InterestRate', 'LoanTerm', 'DTIRatio', 'Education',
    'EmploymentType', 'MaritalStatus', 'HasMortgage', 'HasDependents',
    'LoanPurpose', 'HasCoSigner'
]

# Sidebar - CSV upload or manual
st.sidebar.header("Prediction Options")
mode = st.sidebar.radio("Choose Prediction Mode", ["Single Input", "Upload CSV"])

# Helper function to make prediction
def predict(input_df):
    X_transformed = preprocessor.transform(input_df)
    proba = model.predict_proba(X_transformed)[:, 1]
    return (proba >= 0.5).astype(int), proba

# Main interface
st.title("🏦 Loan Default Prediction App")

if mode == "Single Input":
    st.subheader("Enter Borrower Details")

    input_data = {}
    for feature in FEATURES:
        if feature in ['Education', 'EmploymentType', 'LoanPurpose']:
            input_data[feature] = st.selectbox(f"{feature}", ['High School', 'Bachelors', 'Masters'] if feature == 'Education'
                                               else ['Salaried', 'Self-Employed'] if feature == 'EmploymentType'
                                               else ['Debt Consolidation', 'Home Improvement', 'Credit Card', 'Other'])
        elif feature in ['MaritalStatus', 'HasMortgage', 'HasDependents', 'HasCoSigner']:
            input_data[feature] = st.selectbox(f"{feature}", ['No', 'Yes'])
        else:
            input_data[feature] = st.number_input(f"{feature}", min_value=0.0, step=1.0)

    if st.button("Predict"):
        input_df = pd.DataFrame([input_data])
        pred_class, pred_proba = predict(input_df)
        st.success(f"🔍 Prediction: {'Default' if pred_class[0] == 1 else 'No Default'}")
        st.info(f"Probability of Default: {pred_proba[0]:.2%}")

elif mode == "Upload CSV":
    st.subheader("Upload CSV for Batch Prediction")
    uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)

        if not all(col in df.columns for col in FEATURES):
            st.error(f"CSV must include all required columns:\n{FEATURES}")
        else:
            preds, probas = predict(df)
            result_df = df.copy()
            result_df['Default_Probability'] = probas
            result_df['Prediction'] = np.where(preds == 1, "Default", "No Default")

            st.dataframe(result_df)

            csv_out = result_df.to_csv(index=False).encode("utf-8")
            st.download_button("Download Predictions CSV", csv_out, "predictions.csv", "text/csv")

# Footer
st.markdown("---")
st.markdown("📌 *Model trained with XGBoost + SMOTE + custom preprocessor*")



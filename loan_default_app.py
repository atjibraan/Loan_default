# loan_default_app.py
import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.base import BaseEstimator, TransformerMixin

# ===== Configuration =====
MODEL_PATH = 'loan_default_model.pkl'
PREPROCESSOR_PATH = 'preprocessor.pkl'
DEFAULT_THRESHOLD = 0.2  # Default threshold if none is set

# Feature definitions (same as training)
NUMERICAL_FEATURES = [
    'Age', 'Income', 'LoanAmount', 'CreditScore', 'MonthsEmployed',
    'NumCreditLines', 'InterestRate', 'LoanTerm', 'DTIRatio'
]

CATEGORICAL_FEATURES = ['Education', 'EmploymentType', 'LoanPurpose']
BINARY_FEATURES = ['MaritalStatus', 'HasMortgage', 'HasDependents', 'HasCoSigner']

# Expected column order (without target)
FEATURES_ORDER = [
    'Age', 'Income', 'LoanAmount', 'CreditScore', 'MonthsEmployed',
    'NumCreditLines', 'InterestRate', 'LoanTerm', 'DTIRatio', 'Education',
    'EmploymentType', 'MaritalStatus', 'HasMortgage', 'HasDependents',
    'LoanPurpose', 'HasCoSigner'
]

# ===== Helper Functions =====
@st.cache_resource
def load_artifacts():
    """Load model and preprocessor with caching"""
    artifacts = {
        'model': None,
        'preprocessor': None
    }
    
    try:
        artifacts['model'] = joblib.load(MODEL_PATH)
        artifacts['preprocessor'] = joblib.load(PREPROCESSOR_PATH)
    except Exception as e:
        st.error(f"Error loading artifacts: {str(e)}")
        st.stop()
    
    return artifacts

class Preprocessor(BaseEstimator, TransformerMixin):
    """Replicate preprocessing class from training"""
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

def predict_default_probability(input_data, artifacts):
    """Predict default probability for input data"""
    try:
        # Preprocess input
        processed_data = artifacts['preprocessor'].transform(input_data)
        
        # Predict probability
        probabilities = artifacts['model'].predict_proba(processed_data)
        return probabilities[:, 1]  # Return probability of default (class 1)
    except Exception as e:
        st.error(f"Prediction error: {str(e)}")
        return None

def get_user_input():
    """Collect user input through form"""
    with st.form("loan_input"):
        st.header("Applicant Information")
        
        # Numerical features
        col1, col2, col3 = st.columns(3)
        age = col1.number_input("Age", min_value=18, max_value=100, value=35)
        income = col2.number_input("Annual Income ($)", min_value=1000, value=60000)
        loan_amount = col3.number_input("Loan Amount ($)", min_value=1000, value=20000)
        
        col1, col2, col3 = st.columns(3)
        credit_score = col1.number_input("Credit Score", min_value=300, max_value=850, value=700)
        months_employed = col2.number_input("Months Employed", min_value=0, value=36)
        num_credit_lines = col3.number_input("Number of Credit Lines", min_value=0, value=3)
        
        col1, col2, col3 = st.columns(3)
        interest_rate = col1.number_input("Interest Rate (%)", min_value=0.0, max_value=30.0, value=7.5, step=0.1)
        loan_term = col2.number_input("Loan Term (months)", min_value=1, value=36)
        dti_ratio = col3.number_input("DTI Ratio", min_value=0.0, max_value=1.0, value=0.35, step=0.01)
        
        # Categorical features
        st.subheader("Categorical Information")
        col1, col2, col3 = st.columns(3)
        education = col1.selectbox("Education", ["High School", "Bachelor's", "Master's", "PhD"])
        employment_type = col2.selectbox("Employment Type", ["Full-time", "Part-time", "Self-employed", "Unemployed"])
        loan_purpose = col3.selectbox("Loan Purpose", ["Business", "Home", "Education", "Personal", "Auto"])
        
        # Binary features
        st.subheader("Personal Information")
        col1, col2, col3, col4 = st.columns(4)
        marital_status = col1.selectbox("Marital Status", ["Single", "Married"])
        has_mortgage = col2.selectbox("Has Mortgage", ["No", "Yes"])
        has_dependents = col3.selectbox("Has Dependents", ["No", "Yes"])
        has_cosigner = col4.selectbox("Has Co-signer", ["No", "Yes"])
        
        # Convert binary features to 0/1
        marital_status = 1 if marital_status == "Married" else 0
        has_mortgage = 1 if has_mortgage == "Yes" else 0
        has_dependents = 1 if has_dependents == "Yes" else 0
        has_cosigner = 1 if has_cosigner == "Yes" else 0
        
        submitted = st.form_submit_button("Predict Default Risk")
        
        if submitted:
            return pd.DataFrame([[
                age, income, loan_amount, credit_score, months_employed,
                num_credit_lines, interest_rate, loan_term, dti_ratio,
                education, employment_type, marital_status, has_mortgage,
                has_dependents, loan_purpose, has_cosigner
            ]], columns=FEATURES_ORDER)
    return None

# ===== Main App =====
def main():
    st.set_page_config(
        page_title="Loan Default Predictor",
        page_icon="💰",
        layout="wide"
    )
    
    # Load artifacts
    artifacts = load_artifacts()
    
    # App title
    st.title("Loan Default Risk Assessment")
    st.markdown("""
    This application predicts the probability of a loan applicant defaulting based on their financial profile.
    Adjust the threshold to change risk sensitivity.
    """)
    
    # Threshold selection
    threshold = st.slider(
        "Classification Threshold", 
        min_value=0.0, 
        max_value=1.0, 
        value=DEFAULT_THRESHOLD,
        step=0.01,
        help="Higher values reduce false positives but increase false negatives"
    )
    
    # Get user input
    input_df = get_user_input()
    
    if input_df is not None:
        # Make prediction
        probabilities = predict_default_probability(input_df, artifacts)
        
        if probabilities is not None:
            prob_default = probabilities[0]
            prediction = "High Risk" if prob_default >= threshold else "Low Risk"
            
            # Display results
            st.subheader("Prediction Results")
            
            # Create columns for metrics
            col1, col2 = st.columns(2)
            col1.metric("Default Probability", f"{prob_default:.2%}")
            col2.metric("Risk Classification", prediction)
            
            # Progress bar visualization
            st.progress(prob_default, text=f"Risk Score: {prob_default:.2%}")
            
            # Risk explanation
            if prob_default >= threshold:
                st.warning("⚠️ This applicant is classified as high risk")
                st.markdown("**Recommendation:** Consider additional verification or decline application")
            else:
                st.success("✅ This applicant is classified as low risk")
                st.markdown("**Recommendation:** Application meets risk criteria for approval")
            
            # Threshold explanation
            st.markdown(f"""
            **Threshold Information:**
            - Current classification threshold: {threshold:.0%}
            - Applicant's risk score: {prob_default:.2%}
            - Margin: {(prob_default - threshold):.2%} 
            """)

if __name__ == "__main__":
    # Add necessary imports that are used in the Preprocessor
    from sklearn.compose import ColumnTransformer
    from sklearn.preprocessing import OneHotEncoder, StandardScaler, OrdinalEncoder
    
    main()



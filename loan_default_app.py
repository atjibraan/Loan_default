# loan_default_app.py
import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.base import BaseEstimator, TransformerMixin
import io

# ===== Configuration =====
MODEL_PATH = 'loan_default_model.pkl'
PREPROCESSOR_PATH = 'preprocessor.pkl'
DEFAULT_THRESHOLD = 0.2

# Feature definitions
NUMERICAL_FEATURES = [
    'Age', 'Income', 'LoanAmount', 'CreditScore', 'MonthsEmployed',
    'NumCreditLines', 'InterestRate', 'LoanTerm', 'DTIRatio'
]

CATEGORICAL_FEATURES = ['Education', 'EmploymentType', 'LoanPurpose']
BINARY_FEATURES = ['MaritalStatus', 'HasMortgage', 'HasDependents', 'HasCoSigner']

# Expected column order
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
        # Ensure proper data types
        for col in NUMERICAL_FEATURES:
            input_data[col] = pd.to_numeric(input_data[col], errors='coerce')
        
        # Preprocess input
        processed_data = artifacts['preprocessor'].transform(input_data)
        
        # Predict probability
        probabilities = artifacts['model'].predict_proba(processed_data)
        return probabilities[:, 1]  # Probability of default (class 1)
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
        
        submitted = st.form_submit_button("Predict Default Risk")
        
        if submitted:
            # Create DataFrame with proper data types
            return pd.DataFrame([{
                'Age': int(age),
                'Income': float(income),
                'LoanAmount': float(loan_amount),
                'CreditScore': int(credit_score),
                'MonthsEmployed': int(months_employed),
                'NumCreditLines': int(num_credit_lines),
                'InterestRate': float(interest_rate),
                'LoanTerm': int(loan_term),
                'DTIRatio': float(dti_ratio),
                'Education': education,
                'EmploymentType': employment_type,
                'MaritalStatus': marital_status,
                'HasMortgage': has_mortgage,
                'HasDependents': has_dependents,
                'LoanPurpose': loan_purpose,
                'HasCoSigner': has_cosigner
            }])
    return None

def validate_uploaded_data(df):
    """Validate uploaded CSV data"""
    # Check for required columns
    missing_cols = [col for col in FEATURES_ORDER if col not in df.columns]
    if missing_cols:
        st.error(f"Missing required columns: {', '.join(missing_cols)}")
        return False
    
    # Check data types
    for col in NUMERICAL_FEATURES:
        if not pd.api.types.is_numeric_dtype(df[col]):
            st.error(f"Column '{col}' must be numeric")
            return False
    
    # Check for missing values
    if df.isnull().any().any():
        st.error("Dataset contains missing values")
        return False
    
    return True

def process_batch_data(uploaded_file, artifacts, threshold):
    """Process uploaded CSV file"""
    try:
        df = pd.read_csv(uploaded_file)
        
        # Validate data
        if not validate_uploaded_data(df):
            return None
        
        # Predict probabilities
        probabilities = predict_default_probability(df[FEATURES_ORDER], artifacts)
        
        if probabilities is not None:
            # Add predictions to DataFrame
            df['Default_Probability'] = probabilities
            df['Risk_Classification'] = np.where(
                probabilities >= threshold, 
                "High Risk", 
                "Low Risk"
            )
            return df
    except Exception as e:
        st.error(f"Error processing file: {str(e)}")
    
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
    This application predicts the probability of loan applicants defaulting. 
    Use the single application form or upload a CSV for batch processing.
    """)
    
    # Threshold selection
    threshold = st.sidebar.slider(
        "Classification Threshold", 
        min_value=0.0, 
        max_value=1.0, 
        value=DEFAULT_THRESHOLD,
        step=0.01,
        help="Higher values reduce false positives but increase false negatives"
    )
    
    # Create tabs
    tab1, tab2 = st.tabs(["Single Application", "Batch Processing"])
    
    with tab1:
        st.subheader("Single Application Assessment")
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
                
                # FIXED: Proper progress bar implementation
                st.write(f"Risk Score: {prob_default:.2%}")
                st.progress(float(prob_default))
                
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

    with tab2:
        st.subheader("Batch Processing")
        st.info("Upload a CSV file containing loan applicant data")
        
        # File uploader
        uploaded_file = st.file_uploader(
            "Choose a CSV file", 
            type="csv",
            help="File must contain all required columns in the correct format"
        )
        
        if uploaded_file is not None:
            # Process the uploaded file
            with st.spinner("Processing file..."):
                results_df = process_batch_data(uploaded_file, artifacts, threshold)
            
            if results_df is not None:
                st.success("Predictions completed successfully!")
                
                # Show results
                st.subheader("Prediction Results")
                st.dataframe(results_df)
                
                # Download button
                csv = results_df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="Download Results",
                    data=csv,
                    file_name='loan_predictions.csv',
                    mime='text/csv'
                )

if __name__ == "__main__":
    # Add necessary imports that are used in the Preprocessor
    from sklearn.compose import ColumnTransformer
    from sklearn.preprocessing import OneHotEncoder, StandardScaler, OrdinalEncoder
    
    main()


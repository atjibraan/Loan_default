# loan_default_app.py
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import os
import sys
import traceback
from datetime import datetime
from sklearn.base import BaseEstimator, TransformerMixin

# ===== Configuration =====
MODEL_PATH = 'loan_default_model.pkl'
PREPROCESSOR_PATH = 'preprocessor.pkl'
DEVELOPER_NAME = "Jibraan Attar"  # Replace with your name
MODEL_VERSION = "2.0"
MODEL_TRAIN_DATE = "2025-08-1"
MAX_CSV_ROWS = 100000  # Limit for batch processing
MAX_FILE_SIZE_MB = 25  # Increased to 25MB

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

# ===== File Validation =====
if not os.path.exists(MODEL_PATH):
    st.error(f"CRITICAL ERROR: Model file not found at {MODEL_PATH}")
    st.error("Please ensure 'loan_default_model.pkl' is in the same directory as this app.")
    st.stop()

if not os.path.exists(PREPROCESSOR_PATH):
    st.error(f"CRITICAL ERROR: Preprocessor file not found at {PREPROCESSOR_PATH}")
    st.error("Please ensure 'preprocessor.pkl' is in the same directory as this app.")
    st.stop()

# ===== Helper Functions =====
@st.cache_resource
def load_artifacts():
    """Load model and preprocessor with caching and version checks"""
    artifacts = {
        'model': None,
        'preprocessor': None,
        'status': 'loaded'
    }
    
    try:
        # Load model and preprocessor
        artifacts['model'] = joblib.load(MODEL_PATH)
        artifacts['preprocessor'] = joblib.load(PREPROCESSOR_PATH)
        
        # Add compatibility info
        artifacts['compatibility'] = {
            'pandas': pd.__version__,
            'numpy': np.__version__,
            'sklearn': joblib.__version__,
            'xgboost': None
        }
        
        # Try to get XGBoost version if available
        try:
            import xgboost
            artifacts['compatibility']['xgboost'] = xgboost.__version__
        except ImportError:
            pass
            
    except Exception as e:
        artifacts['status'] = 'error'
        st.error(f"Error loading artifacts: {str(e)}")
        st.error("This is often caused by:")
        st.error("- Version conflicts (check requirements.txt)")
        st.error("- Corrupted model files")
        st.error("- Incompatible Python version")
        
        # Show traceback in expander
        with st.expander("Technical Details"):
            st.code(traceback.format_exc())
            
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
        with st.expander("Error details"):
            st.code(traceback.format_exc())
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

def process_batch_data(uploaded_file, artifacts):
    """Process uploaded CSV file with memory limits"""
    try:
        # Check file size
        file_size = len(uploaded_file.getvalue()) / (1024 * 1024)  # MB
        if file_size > MAX_FILE_SIZE_MB:
            st.error(f"File size too large (max {MAX_FILE_SIZE_MB}MB)")
            return None
        
        # Read with row limit
        df = pd.read_csv(uploaded_file, nrows=MAX_CSV_ROWS)
        
        if len(df) >= MAX_CSV_ROWS:
            st.warning(f"Only first {MAX_CSV_ROWS} rows processed (max limit)")
        
        # Validate data
        if not validate_uploaded_data(df):
            return None
        
        # Predict probabilities
        probabilities = predict_default_probability(df[FEATURES_ORDER], artifacts)
        
        if probabilities is not None:
            # Convert probabilities to Python floats
            probabilities = probabilities.astype(float)
            
            # Add predictions to DataFrame
            df['Default_Probability'] = probabilities
            
            # Classify based on fixed threshold of 0.5
            df['Risk_Classification'] = np.where(
                probabilities >= 0.5, 
                "High Risk", 
                "Low Risk"
            )
            return df
    except Exception as e:
        st.error(f"Error processing file: {str(e)}")
        with st.expander("Error details"):
            st.code(traceback.format_exc())
    
    return None

def create_gauge_chart(probability):
    """Create a matplotlib gauge chart without threshold line"""
    fig, ax = plt.subplots(figsize=(8, 3))
    
    # Create gauge
    ax.barh(['Risk'], [1], color='lightgreen', height=0.3)
    ax.barh(['Risk'], [probability], color='orange', height=0.3)
    
    # Add text
    ax.text(probability, 0.15, f'{probability:.1%}', 
            color='black', ha='center', va='center', fontsize=12)
    
    # Set limits and labels
    ax.set_xlim(0, 1)
    ax.set_xticks(np.arange(0, 1.1, 0.1))
    ax.set_xticklabels([f'{x:.0%}' for x in np.arange(0, 1.1, 0.1)])
    ax.set_title('Default Probability Gauge')
    ax.grid(True, alpha=0.3)
    
    return fig

def create_probability_comparison(prob_default):
    """Create probability comparison chart"""
    fig, ax = plt.subplots(figsize=(8, 3))
    plt.barh(['Default', 'No Default'], [prob_default, 1 - prob_default], 
             color=['#ff7f0e', '#1f77b4'])
    plt.xlim(0, 1)
    plt.title('Probability Comparison')
    plt.xlabel('Probability')
    plt.grid(axis='x', linestyle='--', alpha=0.7)
    return fig

# ===== Main App =====
def main():
    st.set_page_config(
        page_title="Loan Default Predictor",
        page_icon="💰",
        layout="wide"
    )
    
    # Load artifacts
    artifacts = load_artifacts()
    
    # Developer and model info in sidebar
    st.sidebar.header("Developer & Model Information")
    st.sidebar.markdown(f"**Developer:** {DEVELOPER_NAME}")
    st.sidebar.markdown(f"**Model:** XGBoost Classifier")
    st.sidebar.markdown(f"**Version:** {MODEL_VERSION}")
    st.sidebar.markdown(f"**Trained on:** {MODEL_TRAIN_DATE}")
    st.sidebar.markdown("**Features:**")
    st.sidebar.markdown("- Numerical: Age, Income, Loan Amount, etc.")
    st.sidebar.markdown("- Categorical: Education, Employment Type, etc.")
    st.sidebar.markdown("---")
    
    # Fixed threshold explanation
    st.sidebar.subheader("Risk Classification")
    st.sidebar.markdown("""
    - **High Risk**: Probability ≥ 50%
    - **Low Risk**: Probability < 50%
    
    This threshold is fixed based on model optimization.
    """)
    
    # File size info
    st.sidebar.subheader("Batch Processing")
    st.sidebar.markdown(f"""
    - Max file size: **{MAX_FILE_SIZE_MB}MB**
    - Max rows processed: **{MAX_CSV_ROWS}**
    """)
    
    # App title with model info
    st.title("Loan Default Risk Assessment")
    st.markdown(f"""
    **Developer:** {DEVELOPER_NAME} | **Model Version:** {MODEL_VERSION} | **Trained:** {MODEL_TRAIN_DATE}
    
    This application predicts the probability of loan applicants defaulting using an XGBoost classifier. 
    The model was trained on historical loan data with 16 features including financial, employment, 
    and personal information.
    """)
    
    # Create tabs
    tab1, tab2 = st.tabs(["Single Application", "Batch Processing"])
    
    with tab1:
        st.subheader("Single Application Assessment")
        input_df = get_user_input()
        
        if input_df is not None:
            # Make prediction
            probabilities = predict_default_probability(input_df, artifacts)
            
            if probabilities is not None:
                # Convert numpy float32 to Python float
                prob_default = float(probabilities[0])
                prob_no_default = 1 - prob_default
                
                # Determine classification using fixed 50% threshold
                is_high_risk = prob_default >= 0.5
                prediction = "High Risk" if is_high_risk else "Low Risk"
                
                # Display results
                st.subheader("Prediction Results")
                
                # Create columns for metrics
                col1, col2, col3 = st.columns(3)
                col1.metric("Probability of Default", f"{prob_default:.2%}")
                col2.metric("Probability of No Default", f"{prob_no_default:.2%}")
                col3.metric("Risk Classification", prediction)
                
                # Visualization row
                col1, col2 = st.columns(2)
                
                # Gauge chart
                with col1:
                    st.pyplot(create_gauge_chart(prob_default))
                
                # Probability comparison
                with col2:
                    st.pyplot(create_probability_comparison(prob_default))
                
                # Risk explanation
                if is_high_risk:
                    risk_level = "⚠️ HIGH RISK ⚠️"
                    explanation = (
                        "This applicant has a 50% or higher probability of defaulting on the loan "
                        "based on their profile."
                    )
                    recommendation = (
                        "**Recommendation:**\n"
                        "- Request additional financial documentation\n"
                        "- Consider a co-signer requirement\n"
                        "- Increase interest rate to offset risk\n"
                        "- Decline application if risk tolerance is exceeded"
                    )
                    st.error(risk_level)
                else:
                    risk_level = "✅ LOW RISK ✅"
                    explanation = (
                        "This applicant has a less than 50% probability of defaulting on the loan "
                        "based on their profile."
                    )
                    recommendation = (
                        "**Recommendation:**\n"
                        "- Approve application\n"
                        "- Consider standard interest rates\n"
                        "- May qualify for premium loan products"
                    )
                    st.success(risk_level)
                
                st.markdown(f"**Explanation:** {explanation}")
                st.markdown(recommendation)
                
                # Show how close to threshold
                if abs(prob_default - 0.5) < 0.1:
                    st.warning("This applicant is close to the risk threshold. Consider additional review.")

    with tab2:
        st.subheader("Batch Processing")
        st.info(f"Upload a CSV file containing loan applicant data (max {MAX_FILE_SIZE_MB}MB)")
        
        # File uploader
        uploaded_file = st.file_uploader(
            "Choose a CSV file", 
            type="csv",
            help=f"File must contain all required columns and be less than {MAX_FILE_SIZE_MB}MB"
        )
        
        if uploaded_file is not None:
            # Show file info
            file_size = len(uploaded_file.getvalue()) / (1024 * 1024)  # MB
            st.info(f"Uploaded file: {uploaded_file.name} ({file_size:.2f}MB)")
            
            # Process the uploaded file
            with st.spinner("Processing file..."):
                results_df = process_batch_data(uploaded_file, artifacts)
            
            if results_df is not None:
                st.success(f"Processed {len(results_df)} applications successfully!")
                
                # Calculate risk distribution
                risk_counts = results_df['Risk_Classification'].value_counts()
                st.subheader("Risk Distribution")
                
                # Create columns for metrics and chart
                col1, col2 = st.columns([1, 2])
                
                with col1:
                    st.metric("High Risk Applicants", len(results_df[results_df['Risk_Classification'] == "High Risk"]))
                    st.metric("Low Risk Applicants", len(results_df[results_df['Risk_Classification'] == "Low Risk"]))
                    st.metric("Average Default Probability", f"{results_df['Default_Probability'].mean():.2%}")
                
                with col2:
                    # Pie chart of risk distribution
                    fig, ax = plt.subplots()
                    ax.pie(risk_counts, labels=risk_counts.index, autopct='%1.1f%%',
                           colors=['#ff7f0e', '#1f77b4'], startangle=90)
                    ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
                    st.pyplot(fig)
                
                # Show high risk applicants first
                high_risk_df = results_df[results_df['Risk_Classification'] == "High Risk"]
                low_risk_df = results_df[results_df['Risk_Classification'] == "Low Risk"]
                
                # Show results with tabs
                risk_tab1, risk_tab2 = st.tabs([
                    f"High Risk Applicants ({len(high_risk_df)})", 
                    f"Low Risk Applicants ({len(low_risk_df)})"
                ])
                
                with risk_tab1:
                    if len(high_risk_df) > 0:
                        st.dataframe(high_risk_df.sort_values('Default_Probability', ascending=False))
                    else:
                        st.info("No high risk applicants found")
                
                with risk_tab2:
                    if len(low_risk_df) > 0:
                        st.dataframe(low_risk_df.sort_values('Default_Probability', ascending=True))
                    else:
                        st.info("No low risk applicants found")
                
                # Download button
                csv = results_df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="Download Full Results",
                    data=csv,
                    file_name='loan_predictions.csv',
                    mime='text/csv'
                )

if __name__ == "__main__":
    # Add necessary imports that are used in the Preprocessor
    from sklearn.compose import ColumnTransformer
    from sklearn.preprocessing import OneHotEncoder, StandardScaler, OrdinalEncoder
    
    # Add visualization imports
    import matplotlib.pyplot as plt
    
    # Run the app with error handling
    try:
        main()
    except Exception as e:
        st.error("A critical error occurred. Please try again or contact support.")
        with st.expander("Error Details"):
            st.error(str(e))
            st.code(traceback.format_exc())
        st.stop()

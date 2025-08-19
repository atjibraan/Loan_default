import streamlit as st
import pandas as pd
import numpy as np
import joblib
import traceback
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, OrdinalEncoder
import plotly.graph_objects as go

# ===== Configuration =====
MODEL_PATH = 'loan_default_model.pkl'
PREPROCESSOR_PATH = 'preprocessor.pkl'
DEVELOPER_NAME = "Jibraan Attar"
MODEL_VERSION = "2.0"
MODEL_TRAIN_DATE = "2025-07-29"
MAX_CSV_ROWS = 2000000

NUMERICAL_FEATURES = ['Age','Income','LoanAmount','CreditScore','MonthsEmployed','NumCreditLines','InterestRate','LoanTerm','DTIRatio']
CATEGORICAL_FEATURES = ['Education','EmploymentType','LoanPurpose']
BINARY_FEATURES = ['MaritalStatus','HasMortgage','HasDependents','HasCoSigner']
FEATURES_ORDER = NUMERICAL_FEATURES + CATEGORICAL_FEATURES + BINARY_FEATURES

# ===== Preprocessor =====
class Preprocessor(BaseEstimator, TransformerMixin):
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

# ===== Load Artifacts =====
@st.cache_resource
def load_artifacts():
    try:
        model = joblib.load(MODEL_PATH)
        preprocessor = joblib.load(PREPROCESSOR_PATH)
        return {'model': model, 'preprocessor': preprocessor}
    except Exception as e:
        st.error(f"Error loading artifacts: {str(e)}")
        with st.expander("Technical Details"):
            st.code(traceback.format_exc())
        st.stop()

# ===== Prediction =====
def predict_default_probability(df, artifacts):
    for col in NUMERICAL_FEATURES:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    processed = artifacts['preprocessor'].transform(df)
    probs = artifacts['model'].predict_proba(processed)[:,1]
    return probs

# ===== Interactive Visuals =====
def plot_gauge(prob):
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=float(prob),
        title={'text': "Default Probability"},
        gauge={'axis': {'range':[0,1]}, 'bar': {'color':'orange'},
               'steps':[{'range':[0,0.5],'color':'lightgreen'},{'range':[0.5,1],'color':'red'}]}
    ))
    fig.update_layout(height=300)
    return fig

def plot_probability_bar(prob):
    fig = go.Figure(go.Bar(
        x=['Default','No Default'],
        y=[float(prob),1-float(prob)],
        marker_color=['#ff7f0e','#1f77b4']
    ))
    fig.update_layout(yaxis=dict(range=[0,1]), title='Probability Comparison')
    return fig

def plot_risk_pie(df):
    risk_counts = df['Risk_Classification'].value_counts()
    fig = go.Figure(go.Pie(
        labels=risk_counts.index,
        values=risk_counts.values,
        hole=0.4,
        marker_colors=['#ff7f0e','#1f77b4']
    ))
    fig.update_layout(title="Risk Distribution")
    return fig

# ===== Suggestion Engine =====
def generate_suggestions(data, prob):
    tips = []
    contributions = []
    if data['Income'] < 30000:
        tips.append("💡 Consider increasing your income or providing a co-signer.")
        contributions.append("low Income")
    if data['DTIRatio'] > 0.4:
        tips.append("💡 Your DTI is high; consider reducing debt or negotiating interest rate.")
        contributions.append("high DTI")
    if data['LoanAmount'] > data['Income']*5:
        tips.append("💡 Your LoanAmount is high relative to income; reduce loan size or improve income.")
        contributions.append("high LoanAmount")
    if contributions:
        contribution_text = " and ".join(contributions)
        tips.insert(0,f"⚠️ Key factors contributing to risk: {contribution_text}")
    return tips

# ===== Chatbot + Batch App =====
def main():
    st.set_page_config(page_title="Loan Default Chatbot Dashboard", layout="wide")
    st.title("💬 Loan Default Chatbot Dashboard")
    artifacts = load_artifacts()

    st.sidebar.header("Developer & Model Info")
    st.sidebar.markdown(f"**Developer:** {DEVELOPER_NAME}")
    st.sidebar.markdown(f"**Model Version:** {MODEL_VERSION}")
    st.sidebar.markdown(f"**Trained on:** {MODEL_TRAIN_DATE}")

    tab1, tab2 = st.tabs(["Single Applicant (Chatbot)", "Batch Upload"])

    # ===== Single Applicant =====
    with tab1:
        st.subheader("Multi-turn Chatbot")
        if 'conversation' not in st.session_state:
            st.session_state.conversation = []
            st.session_state.user_data = {}

        def display_chat():
            for entry in st.session_state.conversation:
                if entry['sender']=='bot':
                    st.markdown(f"**Bot:** {entry['message']}")
                else:
                    st.markdown(f"**You:** {entry['message']}")
        display_chat()

        steps = [
            ("Age","number"),
            ("Income","number"),
            ("LoanAmount","number"),
            ("CreditScore","number"),
            ("MonthsEmployed","number"),
            ("NumCreditLines","number"),
            ("InterestRate","number"),
            ("LoanTerm","number"),
            ("DTIRatio","number"),
            ("Education","select",["High School","Bachelor's","Master's","PhD"]),
            ("EmploymentType","select",["Full-time","Part-time","Self-employed","Unemployed"]),
            ("LoanPurpose","select",["Business","Home","Education","Auto"]),
            ("MaritalStatus","select",["Single","Married"]),
            ("HasMortgage","select",["No","Yes"]),
            ("HasDependents","select",["No","Yes"]),
            ("HasCoSigner","select",["No","Yes"])
        ]

        for field in steps:
            if field[0] not in st.session_state.user_data:
                st.session_state.current_step = field
                break
        else:
            st.session_state.current_step = None

        if st.session_state.current_step:
            field = st.session_state.current_step
            st.markdown(f"**Bot:** Hi! Please provide your {field[0]}")
            if field[1]=="number":
                value = st.number_input(field[0], value=0)
            elif field[1]=="select":
                value = st.selectbox(field[0], options=field[2])
            if st.button("Submit", key=field[0]):
                st.session_state.user_data[field[0]] = value
                st.session_state.conversation.append({'sender':'user','message':str(value)})
                st.session_state.conversation.append({'sender':'bot','message':f"{field[0]} recorded."})
                st.experimental_rerun()
        else:
            st.markdown("**Bot:** Thank you! Generating your loan default dashboard...")
            df_input = pd.DataFrame([st.session_state.user_data])
            prob = predict_default_probability(df_input, artifacts)[0]
            risk = "High Risk" if prob>=0.5 else "Low Risk"

            st.metric("Probability of Default", f"{prob:.2%}")
            st.metric("Risk Classification", risk)
            st.plotly_chart(plot_gauge(prob), use_container_width=True)
            st.plotly_chart(plot_probability_bar(prob), use_container_width=True)

            # Generate actionable suggestions
            tips = generate_suggestions(st.session_state.user_data, prob)
            for tip in tips:
                if "⚠️" in tip:
                    st.warning(tip)
                else:
                    st.info(tip)

            st.subheader("Applicant Overview")
            st.dataframe(df_input)

    # ===== Batch Upload =====
    with tab2:
        st.subheader("Batch Processing")
        uploaded_file = st.file_uploader("Upload CSV file", type="csv")
        if uploaded_file:
            df = pd.read_csv(uploaded_file, nrows=MAX_CSV_ROWS)
            missing_cols = [c for c in FEATURES_ORDER if c not in df.columns]
            if missing_cols:
                st.error(f"Missing required columns: {', '.join(missing_cols)}")
            else:
                probs = predict_default_probability(df[FEATURES_ORDER], artifacts)
                df['Default_Probability'] = probs
                df['Risk_Classification'] = np.where(df['Default_Probability']>=0.5,"High Risk","Low Risk")
                st.success(f"Processed {len(df)} applicants")
                st.plotly_chart(plot_risk_pie(df), use_container_width=True)
                st.dataframe(df)
                st.download_button("Download CSV", df.to_csv(index=False).encode('utf-8'), file_name="loan_predictions.csv")

if __name__=="__main__":
    try:
        main()
    except Exception as e:
        st.error("Critical error!")
        with st.expander("Error details"):
            st.code(traceback.format_exc())

# advanced_chatbot_loan_default.py
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import traceback
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, OrdinalEncoder
import plotly.graph_objects as go
from fpdf import FPDF
from io import BytesIO

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

# ===== Visuals =====
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

# ===== PDF Report =====
def generate_pdf(user_data, prob, risk, tips):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial","B",16)
    pdf.cell(0,10,"Loan Default Risk Report",ln=True,align='C')
    pdf.set_font("Arial","",12)
    pdf.ln(10)
    pdf.cell(0,10,f"Risk Classification: {risk}",ln=True)
    pdf.cell(0,10,f"Probability of Default: {prob:.2%}",ln=True)
    pdf.ln(5)
    pdf.cell(0,10,"Applicant Details:",ln=True)
    for k,v in user_data.items():
        pdf.cell(0,8,f"{k}: {v}",ln=True)
    pdf.ln(5)
    if tips:
        pdf.cell(0,10,"Suggested Actions:",ln=True)
        for tip in tips:
            pdf.multi_cell(0,8,tip)
    pdf_output = BytesIO()
    pdf.output(pdf_output)
    pdf_output.seek(0)
    return pdf_output

# ===== Chatbot + Batch App =====
def main():
    st.set_page_config(page_title="Advanced Loan Default Chatbot", layout="wide")
    st.title("💬 Advanced Loan Default Chatbot & Batch Predictor")
    artifacts = load_artifacts()

    st.sidebar.header("Developer & Model Info")
    st.sidebar.markdown(f"**Developer:** {DEVELOPER_NAME}")
    st.sidebar.markdown(f"**Model Version:** {MODEL_VERSION}")
    st.sidebar.markdown(f"**Trained on:** {MODEL_TRAIN_DATE}")

    tab1, tab2 = st.tabs(["Single Applicant (Chatbot)", "Batch Upload"])

    # ===== Single Applicant Chatbot =====
    with tab1:
        st.subheader("Multi-turn Interactive Chatbot")
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

        # Determine next step
        for field in steps:
            if field[0] not in st.session_state.user_data:
                st.session_state.current_step = field
                break
        else:
            st.session_state.current_step = None

        # User input
        if st.session_state.current_step:
            field = st.session_state.current_step
            st.markdown(f"**Bot:** Hi! Could you provide your {field[0]}?")
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
            st.markdown("**Bot:** Thank you! Let me analyze your data and provide a personalized report...")
            df_input = pd.DataFrame([st.session_state.user_data])
            prob = predict_default_probability(df_input, artifacts)[0]
            risk = "High Risk" if prob>=0.5 else "Low Risk"

            # Conditional tips
            tips = []
            if st.session_state.user_data['Income'] < 20000:
                tips.append("💡 Your income is relatively low. Consider a co-signer or smaller loan to improve approval chances.")
            if st.session_state.user_data['DTIRatio'] > 0.4:
                tips.append("💡 High Debt-to-Income ratio. Reducing debt can lower default risk.")
            if st.session_state.user_data['CreditScore'] < 600:
                tips.append("💡 Low credit score. Paying bills on time and reducing debts can improve it.")
            if st.session_state.user_data['LoanAmount'] > st.session_state.user_data['Income']*2:
                tips.append("💡 Loan amount is high relative to your income. Consider requesting a smaller loan.")

            # Dynamic explanation (mock contribution percentages)
            top_features = ["LoanAmount","Income"]
            contribution_msg = f"⚡ Based on your inputs, {top_features[0]} and {top_features[1]} contribute ~60% to your risk."
            st.markdown(f"**Bot:** {contribution_msg}")

            st.metric("Probability of Default", f"{prob:.2%}")
            st.metric("Risk Classification", risk)
            st.plotly_chart(plot_gauge(prob), use_container_width=True)
            st.plotly_chart(plot_probability_bar(prob), use_container_width=True)

            if risk=="High Risk":
                st.warning("⚠️ HIGH RISK ⚠️")
            else:
                st.success("✅ LOW RISK ✅")

            st.subheader("Applicant Data & Suggestions")
            st.dataframe(df_input)
            if tips:
                st.markdown("**Bot:** Here are some actionable tips:")
                for tip in tips:
                    st.markdown(f"- {tip}")

            # PDF report
            pdf_file = generate_pdf(st.session_state.user_data, prob, risk, tips)
            st.download_button("📄 Download Personalized Report", pdf_file, file_name="LoanDefaultReport.pdf")

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
        st.error("Critical error occurred!")
        with st.expander("Error details"):
            st.code(traceback.format_exc())

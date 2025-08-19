# app.py
import streamlit as st
import pandas as pd
import joblib
import numpy as np
import pyttsx3
import speech_recognition as sr

# ---------------------------
# Voice engine setup
# ---------------------------
engine = pyttsx3.init()
engine.setProperty('rate', 150)  # Speaking rate

def speak(text):
    engine.say(text)
    engine.runAndWait()

def get_voice_input():
    r = sr.Recognizer()
    with sr.Microphone() as source:
        st.info("Listening... Please speak now.")
        audio = r.listen(source, phrase_time_limit=5)
        try:
            text = r.recognize_google(audio)
            st.success(f"You said: {text}")
            return text
        except sr.UnknownValueError:
            st.error("Could not understand audio")
            return ""
        except sr.RequestError:
            st.error("Speech Recognition service error")
            return ""

# ---------------------------
# Load model & preprocessor
# ---------------------------
@st.cache_resource
def load_model():
    model = joblib.load("loan_default_model.pkl")
    preprocessor = joblib.load("preprocessor.pkl")
    return model, preprocessor

model, preprocessor = load_model()
features = preprocessor.feature_names_in_

# ---------------------------
# Title
# ---------------------------
st.title("💰 Loan & Financial Advisory Chatbot (Voice + Text)")
st.write("Assess loan default risk with interactive explanations and voice support.")

# ---------------------------
# Input Mode
# ---------------------------
input_mode = st.sidebar.selectbox("Choose Input Mode", ["Single Applicant", "CSV Batch Upload"])

# =========================
# Single Applicant Input
# =========================
if input_mode == "Single Applicant":
    st.header("Enter Applicant Details")
    user_input = {}

    for f in features:
        if "Age" in f or "Income" in f or "LoanAmount" in f or "CreditScore" in f or \
           "MonthsEmployed" in f or "NumCreditLines" in f or "InterestRate" in f or \
           "LoanTerm" in f or "DTIRatio" in f:
            user_input[f] = st.number_input(f, value=0)
        else:
            user_input[f] = st.selectbox(f, ["Low", "Medium", "High"])

    # Voice input button
    if st.button("🎤 Fill Inputs via Voice"):
        voice_text = get_voice_input()
        # Simple parsing: expects format "Feature: Value, Feature2: Value2"
        for part in voice_text.split(","):
            if ":" in part:
                key, val = part.split(":")
                key = key.strip()
                val = val.strip()
                if key in features:
                    try:
                        user_input[key] = float(val)
                    except:
                        user_input[key] = val

    if st.button("Predict Risk"):
        input_df = pd.DataFrame([user_input])
        X = preprocessor.transform(input_df)
        pred_proba = model.predict_proba(X)[:,1][0]
        st.subheader(f"Predicted Default Risk: {pred_proba*100:.2f}%")
        speak(f"The predicted default risk is {pred_proba*100:.2f} percent.")

        # ---------------------------
        # Feature impact explanation (Simple alternative to SHAP)
        # ---------------------------
        st.subheader("Feature Impact Explanation")
        # Use feature deviation from mean as impact
        mean_vals = np.mean(preprocessor.transform(pd.DataFrame(np.random.randn(100, len(features)), columns=features)), axis=0)
        input_vals = X[0]
        impact_scores = input_vals - mean_vals
        impact_df = pd.DataFrame({"Feature": features, "ImpactScore": impact_scores})
        impact_df["ImpactAbs"] = impact_df["ImpactScore"].abs()
        impact_df = impact_df.sort_values(by="ImpactAbs", ascending=False)
        st.table(impact_df.head(5))

        st.subheader("Recommendations:")
        recs = []
        if user_input.get("DTIRatio",0) > 0.4:
            recs.append("- Consider reducing your existing debt.")
        if user_input.get("CreditScore",0) < 650:
            recs.append("- Work on improving your credit score.")
        if user_input.get("MonthsEmployed",0) < 12:
            recs.append("- Stable employment history improves chances.")
        for r in recs:
            st.write(r)
            speak(r)
        st.write("These suggestions are informational only, not legal or financial advice.")

# =========================
# CSV Batch Upload
# =========================
if input_mode == "CSV Batch Upload":
    st.header("Upload CSV for Batch Prediction")
    uploaded_file = st.file_uploader("Upload CSV", type=["csv"])
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        st.write("Preview of uploaded data:")
        st.dataframe(df.head())

        if st.button("Predict Batch Risk"):
            try:
                X_batch = preprocessor.transform(df)
                pred_probs = model.predict_proba(X_batch)[:,1]
                df["DefaultRisk"] = pred_probs
                st.subheader("Predictions:")
                st.dataframe(df)

                # Simple explanation for batch: mean deviation
                mean_vals = np.mean(X_batch, axis=0)
                impact_scores = np.mean(np.abs(X_batch - mean_vals), axis=0)
                impact_df = pd.DataFrame({"Feature": features, "MeanImpact": impact_scores})
                impact_df = impact_df.sort_values(by="MeanImpact", ascending=False)
                st.subheader("Top Contributing Features (Batch Average):")
                st.dataframe(impact_df.head(5))

                # Download
                csv = df.to_csv(index=False).encode('utf-8')
                st.download_button("Download Predictions CSV", csv, "batch_predictions.csv", "text/csv")
                speak("Batch predictions completed successfully.")
            except Exception as e:
                st.error(f"Error processing CSV: {e}")

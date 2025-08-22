# 🏦 Loan Default Prediction App

This is a Streamlit web app that predicts the likelihood of a loan applicant defaulting based on key financial and demographic features. The app uses a trained **XGBoost** model along with a **custom preprocessing pipeline** and offers both **manual input** and **CSV batch upload** options.

## 🚀 Demo

Try the app live (hosted on [Streamlit Cloud](https://streamlit.io/)  

🔗Access the web app through https://loandefault-btgh442a3isuenacuailgb.streamlit.app/

---

## 📦 Features

- Predict default risk for individual applicants.
- Upload CSV file for bulk predictions.
- Custom `Preprocessor` pipeline ensures feature alignment.
- Visualizations: prediction output, feature contributions (via SHAP image), and metrics.
- Optimized for consistent deployment with persistent model files (`loan_default_model.pkl`, `preprocessor.pkl`).

---

## 🧠 Model Details

- **Model**: `XGBClassifier`
- **Trained on**: 16 financial and categorical features
- **Custom preprocessing**:
  - `StandardScaler` for numerical data
  - `OneHotEncoder` for categorical data
  - `OrdinalEncoder` for binary data
- **Resampling**: SMOTE + RandomUnderSampler via custom `HybridResampler`

---

## 🧾 Input Features

| Feature           | Description                            |
|------------------|----------------------------------------|
| Age              | Applicant age                          |
| Income           | Monthly income                         |
| LoanAmount       | Total loan amount                      |
| CreditScore      | Credit score (300–850)                 |
| MonthsEmployed   | Employment length                      |
| NumCreditLines   | Number of credit lines                 |
| InterestRate     | Annual interest rate (%)               |
| LoanTerm         | Loan term (in months)                  |
| DTIRatio         | Debt-to-Income Ratio                   |
| Education        | Education level                        |
| EmploymentType   | Type of employment                     |
| MaritalStatus    | Married / Single                       |
| HasMortgage      | Yes / No                               |
| HasDependents    | Yes / No                               |
| LoanPurpose      | Personal / Education / Business / etc. |
| HasCoSigner      | Yes / No                               |

---

## 📁 Files

| File                  | Purpose                                 |
|-----------------------|-----------------------------------------|
| `loan_default_app.py` | Streamlit frontend app                  |
| `loan_default_model.pkl` | Trained XGBoost model                |
| `preprocessor.pkl`    | Preprocessing pipeline object           |
| `sample_input.csv`    | Example CSV for batch prediction        |
| `shap_plot.png`       | SHAP summary plot for feature importances |
| `requirements.txt`    | Required dependencies                   |

---

## ▶️ How to Run Locally

1. **Clone the repo**:
   ```bash
   git clone https://github.com/your-username/loan-default-app.git
   cd loan-default-app

import streamlit as st
import pandas as pd
import numpy as np
import os
import json
import xgboost as xgb

# --- PAGE CONFIG ---
st.set_page_config(page_title="Zanzibar Microfinance Risk Engine", layout="wide")

# --- LOAD MODEL ---
@st.cache_resource
def load_model():
    if not os.path.exists('xgb_credit_model.json') or not os.path.exists('feature_columns.json'):
        st.error("JSON model files not found! Please ensure the .json files are in this folder.")
        st.stop()
    
    model = xgb.XGBClassifier()
    model.load_model('xgb_credit_model.json')
    
    with open('feature_columns.json', 'r') as f:
        cols = json.load(f)
        
    return model, cols

model, feature_cols = load_model()

# --- FINANCIAL ASSUMPTIONS ---
LOAN_SIZE_TZS = 1_000_000
EXCHANGE_RATE = 2600 # Approximate TZS to USD rate

# --- UI LAYOUT ---
st.title("🏦 Zanzibar Microfinance Risk Engine")
st.markdown("Data-driven loan approval for SACCOs and informal lenders. Replace gut-feel with predictive analytics.")

# Sidebar for ROI Tracking
with st.sidebar:
    st.header("📊 Session ROI Tracker")
    if 'bad_loans_avoided' not in st.session_state:
        st.session_state.bad_loans_avoided = 0
    if 'good_loans_approved' not in st.session_state:
        st.session_state.good_loans_approved = 0
        
    st.metric("Bad Loans Avoided", f"{st.session_state.bad_loans_avoided}")
    st.metric("Good Loans Approved", f"{st.session_state.good_loans_approved}")
    
    money_saved = st.session_state.bad_loans_avoided * LOAN_SIZE_TZS
    st.success(f"💰 Capital Saved: {money_saved:,.0f} TZS")

# --- MAIN INPUT FORM ---
st.subheader("Applicant Assessment")
col1, col2 = st.columns(2)

with col1:
    st.markdown("**1. Business & Income**")
    age = st.number_input("Applicant Age:", min_value=18, max_value=100, value=30)
    biz_tenure = st.slider("Years operating business/duka:", 0, 60, 3)
    monthly_income_tzs = st.number_input("Average monthly income (TZS):", min_value=100000, value=500000, step=50000)
    loan_amount_tzs = st.number_input("Requested loan amount (TZS):", min_value=100000, value=1000000, step=100000)
    interest_rate = st.slider("Proposed Monthly Interest Rate (%):", 5, 25, 10)

with col2:
    st.markdown("**2. Profile & History**")
    housing = st.selectbox("Housing status:", ["Rent", "Own", "Mortgage", "Other"])
    loan_intent = st.selectbox("Purpose of loan:", [
        "Personal", "Education", "Home Improvement", "Medical", "Venture", "Debt Consolidation"
    ])
    past_default = st.radio("Has applicant ever defaulted on a VICOBA or local lender loan?", ["No", "Yes"])
    credit_history_years = st.slider("Years of borrowing history (VICOBA, banks, etc.):", 0, 30, 5)

# --- DATA TRANSLATION (Mapping UI to Kaggle Features) ---
def map_to_kaggle_features():
    # 1. Convert TZS to USD for the model
    annual_income_usd = (monthly_income_tzs * 12) / EXCHANGE_RATE
    loan_amnt_usd = loan_amount_tzs / EXCHANGE_RATE
    
    # 2. Calculate loan percent income
    loan_pct_income = (loan_amnt_usd / annual_income_usd) if annual_income_usd > 0 else 1.0
    
    # 3. Map categories to exact column names
    housing_map = {'Rent': 'RENT', 'Own': 'OWN', 'Mortgage': 'MORTGAGE', 'Other': 'OTHER'}
    intent_map = {
        "Personal": "PERSONAL", "Education": "EDUCATION", "Home Improvement": "HOMEIMPROVEMENT", 
        "Medical": "MEDICAL", "Venture": "VENTURE", "Debt Consolidation": "DEBTCONSOLIDATION"
    }
    
    # 4. Create a base dictionary with all 25 features set to 0
    data = {col: 0 for col in feature_cols}
    
    # 5. Fill in the continuous variables
    data['person_age'] = age
    data['person_income'] = annual_income_usd
    data['person_emp_length'] = biz_tenure
    data['loan_amnt'] = loan_amnt_usd
    data['loan_int_rate'] = interest_rate / 100.0 
    data['loan_percent_income'] = loan_pct_income
    data['cb_person_default_on_file'] = 1 if past_default == "Yes" else 0
    data['cb_person_cred_hist_length'] = credit_history_years
    
    # 6. Fill in the one-hot encoded categorical variables
    data[f'person_home_ownership_{housing_map[housing]}'] = 1
    data[f'loan_intent_{intent_map[loan_intent]}'] = 1
    
    # 7. Proxy for Loan Grade (A-G) based on risk profile
    if interest_rate <= 10 and past_default == "No":
        grade = 'A'
    elif interest_rate <= 12:
        grade = 'B'
    elif interest_rate <= 15:
        grade = 'C'
    elif interest_rate <= 18:
        grade = 'D'
    elif interest_rate <= 21:
        grade = 'E'
    elif interest_rate <= 24:
        grade = 'F'
    else:
        grade = 'G'
        
    data[f'loan_grade_{grade}'] = 1

    return pd.DataFrame([data])

# --- PREDICTION & OUTPUT ---
if st.button("Evaluate Applicant", type="primary"):
    input_data = map_to_kaggle_features()
    
    # Ensure columns are in the exact same order as training
    input_data = input_data[feature_cols]
    
    # Predict probability of default
    prob_default = model.predict_proba(input_data)[0][1]
    risk_score = prob_default * 100
    
    # Decision threshold
    threshold = 35.0 
    is_approved = risk_score < threshold
    
    st.divider()
    
    # Display Results
    col_res1, col_res2 = st.columns(2)
    with col_res1:
        st.metric("Default Probability", f"{risk_score:.1f}%")
    with col_res2:
        if is_approved:
            st.success("✅ RECOMMENDATION: APPROVE")
            st.session_state.good_loans_approved += 1
        else:
            st.error("❌ RECOMMENDATION: REJECT")
            st.session_state.bad_loans_avoided += 1
            
    # Explainability
    st.subheader("🧠 Why did the model decide this?")
    if risk_score > 50:
        st.warning("High risk primarily driven by: **High Debt-to-Income Ratio** and/or **Past VICOBA/Local Defaults**.")
    elif risk_score > 35:
        st.info("Moderate risk. The requested loan amount is a significant percentage of the applicant's annual income.")
    else:
        st.success("Low risk. Applicant has stable business tenure, low debt-to-income ratio, and no past defaults.")
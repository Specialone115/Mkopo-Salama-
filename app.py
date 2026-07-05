import streamlit as st
import pandas as pd
import numpy as np
import os
import json
import xgboost as xgb
import plotly.graph_objects as go
import time

# --- CUSTOM CSS: TANZANIAN FLAG THEME (Green + Gold) ---
st.markdown("""
<style>
    /* Import professional fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700;800&display=swap');
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Main theme colors */
    :root {
        --tz-green: #1EB53A;
        --tz-gold: #FCD116;
        --tz-blue: #00A3DD;
        --dark-bg: #0a1628;
        --card-bg: #ffffff;
        --text-dark: #1a202c;
        --text-light: #4a5568;
    }
    
    /* Global font */
    * {
        font-family: 'Inter', sans-serif !important;
    }
    
    /* Main app background */
    .stApp {
        background: linear-gradient(135deg, #f7fafc 0%, #edf2f7 100%);
    }
    
    /* Hero header with Tanzanian colors */
    .hero-container {
        background: linear-gradient(135deg, #1EB53A 0%, #0a8f2d 100%);
        border-radius: 20px;
        padding: 3rem 2rem;
        margin-bottom: 2rem;
        box-shadow: 0 10px 40px rgba(30, 181, 58, 0.3);
        position: relative;
        overflow: hidden;
    }
    
    .hero-container::before {
        content: '';
        position: absolute;
        top: 0;
        right: 0;
        width: 300px;
        height: 300px;
        background: radial-gradient(circle, rgba(252, 209, 22, 0.3) 0%, transparent 70%);
        border-radius: 50%;
    }
    
    .hero-container::after {
        content: '';
        position: absolute;
        bottom: -50px;
        left: -50px;
        width: 200px;
        height: 200px;
        background: radial-gradient(circle, rgba(0, 163, 221, 0.2) 0%, transparent 70%);
        border-radius: 50%;
    }
    
    .hero-title {
        color: white !important;
        font-size: 3.5rem !important;
        font-weight: 800 !important;
        margin: 0 !important;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.2);
        position: relative;
        z-index: 1;
    }
    
    .hero-icon {
        font-size: 4rem;
        margin-bottom: 1rem;
        display: inline-block;
        animation: float 3s ease-in-out infinite;
    }
    
    @keyframes float {
        0%, 100% { transform: translateY(0px); }
        50% { transform: translateY(-10px); }
    }
    
    .hero-subtitle {
        color: rgba(255, 255, 255, 0.95) !important;
        font-size: 1.3rem !important;
        margin-top: 0.5rem !important;
        font-weight: 400 !important;
        position: relative;
        z-index: 1;
    }
    
    .hero-tagline {
        display: inline-block;
        background: rgba(252, 209, 22, 0.9);
        color: #1a202c !important;
        padding: 0.6rem 1.5rem;
        border-radius: 50px;
        margin-top: 1.5rem;
        font-weight: 700 !important;
        font-size: 1rem !important;
        box-shadow: 0 4px 15px rgba(252, 209, 22, 0.4);
        position: relative;
        z-index: 1;
    }
    
    /* Section headers */
    .section-title {
        background: linear-gradient(135deg, #1EB53A 0%, #0a8f2d 100%);
        color: white !important;
        padding: 1rem 1.5rem;
        border-radius: 10px;
        font-size: 1.3rem !important;
        font-weight: 700 !important;
        margin: 2rem 0 1.5rem 0 !important;
        box-shadow: 0 4px 15px rgba(30, 181, 58, 0.3);
    }
    
    /* Metric cards in sidebar */
    .metric-card {
        background: linear-gradient(135deg, #ffffff 0%, #f7fafc 100%);
        border-radius: 12px;
        padding: 1.5rem;
        margin-bottom: 1rem;
        border-left: 4px solid #1EB53A;
        box-shadow: 0 2px 10px rgba(0, 0, 0, 0.05);
    }
    
    .metric-label {
        color: #4a5568;
        font-size: 0.85rem;
        text-transform: uppercase;
        letter-spacing: 1px;
        font-weight: 600;
        margin-bottom: 0.5rem;
    }
    
    .metric-value {
        color: #1a202c;
        font-size: 2.5rem;
        font-weight: 800;
        line-height: 1;
    }
    
    /* Result cards */
    .result-card {
        border-radius: 20px;
        padding: 3rem 2rem;
        text-align: center;
        box-shadow: 0 10px 40px rgba(0, 0, 0, 0.15);
        position: relative;
        overflow: hidden;
    }
    
    .result-approve {
        background: linear-gradient(135deg, #1EB53A 0%, #0a8f2d 100%);
        border: 3px solid #FCD116;
    }
    
    .result-reject {
        background: linear-gradient(135deg, #e53e3e 0%, #c53030 100%);
        border: 3px solid #FCD116;
    }
    
    .result-icon {
        font-size: 5rem;
        margin-bottom: 1rem;
        animation: bounce 1s ease-in-out;
    }
    
    @keyframes bounce {
        0%, 100% { transform: translateY(0); }
        50% { transform: translateY(-20px); }
    }
    
    .result-title {
        font-size: 2.5rem;
        font-weight: 800;
        color: white;
        margin-bottom: 1rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
    }
    
    .result-subtitle {
        font-size: 1.1rem;
        color: rgba(255, 255, 255, 0.95);
        font-weight: 500;
    }
    
    /* Info boxes */
    .info-box {
        background: #ebf8ff;
        border-left: 4px solid #00A3DD;
        border-radius: 10px;
        padding: 1.5rem;
        margin: 1.5rem 0;
        color: #1a202c;
    }
    
    .warning-box {
        background: #fff5f5;
        border-left: 4px solid #e53e3e;
        border-radius: 10px;
        padding: 1.5rem;
        margin: 1.5rem 0;
        color: #1a202c;
    }
    
    .success-box {
        background: #f0fff4;
        border-left: 4px solid #1EB53A;
        border-radius: 10px;
        padding: 1.5rem;
        margin: 1.5rem 0;
        color: #1a202c;
    }
    
    /* Financial impact card */
    .impact-card {
        background: linear-gradient(135deg, #FCD116 0%, #f6b800 100%);
        border-radius: 15px;
        padding: 2rem;
        text-align: center;
        box-shadow: 0 8px 30px rgba(252, 209, 22, 0.4);
        border: 2px solid #1EB53A;
    }
    
    .impact-value {
        font-size: 3rem;
        font-weight: 900;
        color: #1a202c;
        margin: 1rem 0;
    }
    
    /* Sidebar styling - FIXED FOR VISIBILITY */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0a1628 0%, #1a202c 100%);
    }
    
    [data-testid="stSidebar"] .stMarkdown,
    [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] .stRadio label,
    [data-testid="stSidebar"] .stSelectbox label,
    [data-testid="stSidebar"] .stNumberInput label {
        color: white !important;
    }
    
    [data-testid="stSidebar"] h2 {
        color: #FCD116 !important;
    }
    
    /* Main content text - ENSURE VISIBILITY */
    .stMarkdown p,
    .stMarkdown li,
    .stMarkdown h1,
    .stMarkdown h2,
    .stMarkdown h3,
    .stMarkdown h4 {
        color: #1a202c !important;
    }
    
    /* Input labels */
    label {
        color: #1a202c !important;
        font-weight: 600 !important;
    }
    
    /* Primary button */
    .stButton > button {
        background: linear-gradient(135deg, #1EB53A 0%, #0a8f2d 100%) !important;
        border: 2px solid #FCD116 !important;
        border-radius: 50px !important;
        padding: 1rem 3rem !important;
        font-size: 1.2rem !important;
        font-weight: 700 !important;
        color: white !important;
        box-shadow: 0 4px 20px rgba(30, 181, 58, 0.4) !important;
        transition: all 0.3s ease !important;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 6px 30px rgba(30, 181, 58, 0.6) !important;
    }
    
    /* Input fields */
    .stNumberInput input, 
    .stSelectbox select,
    .stTextInput input {
        border-radius: 10px !important;
        border: 2px solid #e2e8f0 !important;
        color: #1a202c !important;
        background: white !important;
    }
    
    .stNumberInput input:focus, 
    .stSelectbox select:focus,
    .stTextInput input:focus {
        border-color: #1EB53A !important;
        box-shadow: 0 0 0 3px rgba(30, 181, 58, 0.1) !important;
    }
    
    /* Radio buttons */
    .stRadio > div {
        color: #1a202c !important;
    }
    
    /* Spinner */
    .stSpinner > div {
        border-top-color: #1EB53A !important;
    }
</style>
""", unsafe_allow_html=True)

# --- PAGE CONFIG ---
st.set_page_config(
    page_title="Mkopo Salama - Smart Loan Decisions",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- LOAD MODEL ---
@st.cache_resource
def load_model():
    if not os.path.exists('xgb_credit_model.json') or not os.path.exists('feature_columns.json'):
        st.error("⚠️ Model files not found!")
        st.stop()
    
    model = xgb.XGBClassifier()
    model.load_model('xgb_credit_model.json')
    
    with open('feature_columns.json', 'r') as f:
        cols = json.load(f)
        
    return model, cols

model, feature_cols = load_model()

# --- FINANCIAL ASSUMPTIONS ---
LOAN_SIZE_TZS = 1_000_000
EXCHANGE_RATE = 2600

# --- HERO HEADER ---
st.markdown("""
<div class="hero-container">
    <div class="hero-icon">🏦</div>
    <h1 class="hero-title">Mkopo Salama</h1>
    <p class="hero-subtitle">AI-Powered Credit Risk Assessment for Tanzanian Microfinance</p>
    <div class="hero-tagline">✨ Transform Gut-Feel into Data-Driven Decisions</div>
</div>
""", unsafe_allow_html=True)

# --- SIDEBAR: IMPACT DASHBOARD ---
with st.sidebar:
    st.markdown("## 📊 Real-Time Impact")
    st.markdown("---")
    
    if 'bad_loans_avoided' not in st.session_state:
        st.session_state.bad_loans_avoided = 0
    if 'good_loans_approved' not in st.session_state:
        st.session_state.good_loans_approved = 0
    if 'total_evaluated' not in st.session_state:
        st.session_state.total_evaluated = 0
        
    st.session_state.total_evaluated = st.session_state.bad_loans_avoided + st.session_state.good_loans_approved
    money_saved = st.session_state.bad_loans_avoided * LOAN_SIZE_TZS
    
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Total Evaluated</div>
        <div class="metric-value">{st.session_state.total_evaluated}</div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">🛡️ Bad Loans Avoided</div>
        <div class="metric-value" style="color: #1EB53A;">{st.session_state.bad_loans_avoided}</div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">✅ Good Loans Approved</div>
        <div class="metric-value" style="color: #00A3DD;">{st.session_state.good_loans_approved}</div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    st.markdown(f"""
    <div class="impact-card">
        <div style="font-size: 1rem; font-weight: 600; color: #1a202c;">💰 Capital Protected</div>
        <div class="impact-value">{money_saved:,.0f}</div>
        <div style="font-size: 1.2rem; font-weight: 700; color: #1a202c;">TZS</div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; padding: 1rem; color: rgba(255,255,255,0.7); font-size: 0.9rem;">
        <strong style="color: #FCD116;">Powered by AI</strong><br>
        XGBoost Machine Learning<br>
        Trained on 32,588 loans
    </div>
    """, unsafe_allow_html=True)

# --- MAIN INPUT FORM ---
st.markdown('<div class="section-title">📝 Applicant Assessment</div>', unsafe_allow_html=True)

with st.container():
    col1, col2 = st.columns(2)
    
    with col1:
        with st.container():
            st.markdown("### 💼 Business & Financial Profile")
            age = st.number_input("Applicant Age", min_value=18, max_value=100, value=30)
            biz_tenure = st.slider("Years in Business", 0, 60, 3)
            monthly_income_tzs = st.number_input("Monthly Income (TZS)", min_value=100000, value=500000, step=50000)
            loan_amount_tzs = st.number_input("Loan Amount Requested (TZS)", min_value=100000, value=1000000, step=100000)
            interest_rate = st.slider("Proposed Interest Rate (%)", 5, 25, 10)
    
    with col2:
        with st.container():
            st.markdown("### 🏠 Personal Profile & History")
            housing = st.selectbox("Housing Status", ["Rent", "Own", "Mortgage", "Other"])
            loan_intent = st.selectbox("Loan Purpose", [
                "Personal", "Education", "Home Improvement", "Medical", "Venture", "Debt Consolidation"
            ])
            past_default = st.radio("Previous Default History", ["No", "Yes"])
            credit_history_years = st.slider("Credit History (Years)", 0, 30, 5)

st.markdown("---")

# --- DATA TRANSLATION FUNCTION ---
def map_to_kaggle_features():
    annual_income_usd = (monthly_income_tzs * 12) / EXCHANGE_RATE
    loan_amnt_usd = loan_amount_tzs / EXCHANGE_RATE
    loan_pct_income = (loan_amnt_usd / annual_income_usd) if annual_income_usd > 0 else 1.0
    
    housing_map = {'Rent': 'RENT', 'Own': 'OWN', 'Mortgage': 'MORTGAGE', 'Other': 'OTHER'}
    intent_map = {
        "Personal": "PERSONAL", "Education": "EDUCATION", "Home Improvement": "HOMEIMPROVEMENT", 
        "Medical": "MEDICAL", "Venture": "VENTURE", "Debt Consolidation": "DEBTCONSOLIDATION"
    }
    
    data = {col: 0 for col in feature_cols}
    
    data['person_age'] = age
    data['person_income'] = annual_income_usd
    data['person_emp_length'] = biz_tenure
    data['loan_amnt'] = loan_amnt_usd
    data['loan_int_rate'] = interest_rate / 100.0 
    data['loan_percent_income'] = loan_pct_income
    data['cb_person_default_on_file'] = 1 if past_default == "Yes" else 0
    data['cb_person_cred_hist_length'] = credit_history_years
    
    data[f'person_home_ownership_{housing_map[housing]}'] = 1
    data[f'loan_intent_{intent_map[loan_intent]}'] = 1
    
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

# --- PREDICTION & VISUAL OUTPUT ---
if st.button("🔍 Evaluate Applicant", type="primary", use_container_width=True):
    with st.spinner("🧠 AI is analyzing the applicant profile..."):
        time.sleep(1.5)
        
        input_data = map_to_kaggle_features()
        input_data = input_data[feature_cols]
        
        prob_default = model.predict_proba(input_data)[0][1]
        risk_score = prob_default * 100
        
        threshold = 35.0 
        is_approved = risk_score < threshold
        
        # Get feature importance for explainability
        feature_importance = model.feature_importances_
        feature_names = feature_cols
        importance_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': feature_importance
        }).sort_values('Importance', ascending=False).head(10)
        
        st.markdown("---")
        st.markdown('<div class="section-title">📊 Risk Assessment Result</div>', unsafe_allow_html=True)
        
        # Display results with gauge and decision
        col_gauge, col_result = st.columns([1, 2])
        
        with col_gauge:
            if risk_score < 20:
                risk_color = "#1EB53A"
                risk_level = "LOW"
            elif risk_score < 50:
                risk_color = "#FCD116"
                risk_level = "MEDIUM"
            else:
                risk_color = "#e53e3e"
                risk_level = "HIGH"
            
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=risk_score,
                number={'suffix': "%", 'font': {'size': 40, 'color': risk_color}},
                gauge={
                    'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "#1a202c"},
                    'bar': {'color': risk_color},
                    'bgcolor': "white",
                    'borderwidth': 2,
                    'bordercolor': "#e2e8f0",
                    'steps': [
                        {'range': [0, 20], 'color': '#d4edda'},
                        {'range': [20, 50], 'color': '#fff3cd'},
                        {'range': [50, 100], 'color': '#f8d7da'}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': threshold
                    }
                }
            ))
            
            fig.update_layout(
                height=300,
                margin=dict(l=20, r=20, t=50, b=20),
                paper_bgcolor="white",
                font={'color': "#1a202c", 'family': "Inter"}
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            st.markdown(f"""
            <div style="text-align: center; margin-top: 1rem;">
                <span style="background: {risk_color}; color: white; padding: 0.7rem 2rem; border-radius: 50px; font-weight: 700; font-size: 1.3rem; box-shadow: 0 4px 15px {risk_color}40;">
                    {risk_level} RISK
                </span>
            </div>
            """, unsafe_allow_html=True)
        
        with col_result:
            if is_approved:
                st.markdown(f"""
                <div class="result-card result-approve">
                    <div class="result-icon">✅</div>
                    <div class="result-title">APPROVE</div>
                    <div class="result-subtitle">This applicant meets the risk criteria for loan approval</div>
                </div>
                """, unsafe_allow_html=True)
                st.session_state.good_loans_approved += 1
                st.balloons()
            else:
                st.markdown(f"""
                <div class="result-card result-reject">
                    <div class="result-icon">❌</div>
                    <div class="result-title">REJECT</div>
                    <div class="result-subtitle">This applicant exceeds acceptable risk thresholds</div>
                </div>
                """, unsafe_allow_html=True)
                st.session_state.bad_loans_avoided += 1
        
        st.markdown("---")
        
        st.markdown("### 🧠 Why Did the AI Make This Decision?")
        
        col_explain, col_chart = st.columns([1, 1])
        
        with col_explain:
            if risk_score > 50:
                st.markdown(f"""
                <div class="warning-box">
                    <h4 style="color: #e53e3e; margin-top: 0;">⚠️ High Risk Factors</h4>
                    <ul style="color: #1a202c; font-size: 1rem; line-height: 1.8;">
                        <li><strong>High Debt-to-Income Ratio</strong></li>
                        <li><strong>Past Default History</strong></li>
                        <li><strong>Recommendation:</strong> Require additional collateral</li>
                    </ul>
                </div>
                """, unsafe_allow_html=True)
            elif risk_score > 35:
                st.markdown(f"""
                <div class="info-box">
                    <h4 style="color: #00A3DD; margin-top: 0;">ℹ️ Moderate Risk Factors</h4>
                    <ul style="color: #1a202c; font-size: 1rem; line-height: 1.8;">
                        <li><strong>Elevated Loan-to-Income Ratio</strong></li>
                        <li><strong>Recommendation:</strong> Consider smaller loan amount</li>
                    </ul>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="success-box">
                    <h4 style="color: #1EB53A; margin-top: 0;">✅ Low Risk Profile</h4>
                    <ul style="color: #1a202c; font-size: 1rem; line-height: 1.8;">
                        <li><strong>Stable Business Tenure</strong></li>
                        <li><strong>Healthy Debt-to-Income Ratio</strong></li>
                        <li><strong>Clean Credit History</strong></li>
                    </ul>
                </div>
                """, unsafe_allow_html=True)
        
        with col_chart:
            fig_importance = go.Figure(go.Bar(
                x=importance_df['Importance'],
                y=importance_df['Feature'],
                orientation='h',
                marker=dict(
                    color='rgba(30, 181, 58, 0.8)',
                    line=dict(color='rgba(30, 181, 58, 1)', width=2)
                )
            ))
            
            fig_importance.update_layout(
                title="Top 10 Risk Factors",
                xaxis_title="Importance",
                yaxis_title="Feature",
                height=400,
                margin=dict(l=20, r=20, t=50, b=20),
                paper_bgcolor="white",
                plot_bgcolor="white",
                font={'family': "Inter", 'size': 12}
            )
            
            st.plotly_chart(fig_importance, use_container_width=True)
        
        st.markdown("---")
        st.markdown("### 💰 Financial Impact")
        
        if is_approved:
            potential_profit = LOAN_SIZE_TZS * 0.05 * 6
            st.markdown(f"""
            <div class="impact-card">
                <div style="font-size: 1.2rem; font-weight: 600; color: #1a202c;">Potential Revenue from This Loan</div>
                <div class="impact-value">{potential_profit:,.0f} TZS</div>
                <div style="font-size: 1rem; color: #1a202c; opacity: 0.85;">Based on {LOAN_SIZE_TZS:,.0f} TZS at 5% monthly for 6 months</div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="impact-card">
                <div style="font-size: 1.2rem; font-weight: 600; color: #1a202c;">Capital Protected by This Decision</div>
                <div class="impact-value">{LOAN_SIZE_TZS:,.0f} TZS</div>
                <div style="font-size: 1rem; color: #1a202c; opacity: 0.85;">Avoided potential loss from loan default</div>
            </div>
            """, unsafe_allow_html=True)

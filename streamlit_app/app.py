"""
Streamlit application for Fraud Detection serving interface
Connects to the FastAPI backend API and provides scenario selectors
"""

import streamlit as st
import requests
import json

# Set Page Config
st.set_page_config(
    page_title="Fraud Detection Serving Portal",
    page_icon="🛡️",
    layout="wide"
)

# Colors and Custom CSS
st.markdown("""
<style>
    .big-title {
        font-size: 2.5rem;
        color: #1E3A8A;
        font-weight: bold;
        text-align: center;
        margin-bottom: 1rem;
    }
    .subtitle {
        font-size: 1.1rem;
        color: #4B5563;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-box {
        background-color: #F3F4F6;
        border-radius: 8px;
        padding: 1.5rem;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="big-title">🛡️ Fraud Detection MLOps Interface</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Real-Time XGBoost Scoring Engine Portal (MLSD Academic Showcase)</div>', unsafe_allow_html=True)

# API Server URL
API_URL = "http://localhost:8000"

# Main Layout
col1, col2 = st.columns([2, 1])

# Initial Form State dict
scenario_data = None

with col2:
    st.subheader("⚡ Quick Scenarios")
    st.write("Trigger quick presets to test the decision boundary:")
    
    col_s1, col_s2 = st.columns(2)
    with col_s1:
        if st.button("✅ Legitimate Buy"):
            scenario_data = {
                "TransactionAmt": 45.50,
                "card1": 1000,
                "card2": 100.0,
                "card3": 100.0,
                "card5": 200.0,
                "addr1": 300.0,
                "addr2": 87.0,
                "dist1": 12.0,
                "dist2": -999.0
            }
            st.success("Loaded legitimate scenario!")
            
    with col_s2:
        if st.button("🚨 Suspicious Cashout"):
            scenario_data = {
                "TransactionAmt": 9999.99,
                "card1": 9999,
                "card2": 999.0,
                "card3": 999.0,
                "card5": 999.0,
                "addr1": 999.0,
                "addr2": 999.0,
                "dist1": 50000.0,
                "dist2": 99999.0
            }
            st.error("Loaded suspicious scenario!")

# Inputs panel
with col1:
    st.subheader("📝 Transaction Features")
    
    # Check if a preset scenario was loaded
    default_amt = scenario_data["TransactionAmt"] if scenario_data else 150.0
    default_c1 = scenario_data["card1"] if scenario_data else 5000
    default_c2 = scenario_data["card2"] if scenario_data else 150.0
    default_c3 = scenario_data["card3"] if scenario_data else 150.0
    default_c5 = scenario_data["card5"] if scenario_data else 226.0
    default_a1 = scenario_data["addr1"] if scenario_data else 299.0
    default_a2 = scenario_data["addr2"] if scenario_data else 87.0
    default_d1 = scenario_data["dist1"] if scenario_data else 50.0
    default_d2 = scenario_data["dist2"] if scenario_data else -999.0

    form_amt = st.number_input("Transaction Amount (USD)", min_value=0.1, max_value=100000.0, value=default_amt, step=10.0)
    
    col_c1, col_c2, col_c3 = st.columns(3)
    with col_c1:
        form_c1 = st.number_input("Card Parameter 1 (card1)", min_value=1, max_value=20000, value=int(default_c1))
    with col_c2:
        form_c2 = st.number_input("Card Parameter 2 (card2)", min_value=-999.0, max_value=1000.0, value=default_c2)
    with col_c3:
        form_c3 = st.number_input("Card Parameter 3 (card3)", min_value=-999.0, max_value=1000.0, value=default_c3)
        
    col_c5, col_a1, col_a2 = st.columns(3)
    with col_c5:
        form_c5 = st.number_input("Card Parameter 5 (card5)", min_value=-999.0, max_value=1000.0, value=default_c5)
    with col_a1:
        form_a1 = st.number_input("Billing Zip Area (addr1)", min_value=-999.0, max_value=1000.0, value=default_a1)
    with col_a2:
        form_a2 = st.number_input("Billing Country Code (addr2)", min_value=-999.0, max_value=1000.0, value=default_a2)
        
    col_d1, col_d2 = st.columns(2)
    with col_d1:
        form_d1 = st.number_input("Address Distance 1 (dist1)", min_value=-999.0, max_value=100000.0, value=default_d1)
    with col_d2:
        form_d2 = st.number_input("Address Distance 2 (dist2)", min_value=-999.0, max_value=100000.0, value=default_d2)

# Prediction execution
if st.button("🔍 Analyze Transaction for Fraud Risk", type="primary", use_container_width=True):
    payload = {
        "TransactionAmt": form_amt,
        "card1": form_c1,
        "card2": form_c2,
        "card3": form_c3,
        "card5": form_c5,
        "addr1": form_a1,
        "addr2": form_a2,
        "dist1": form_d1,
        "dist2": form_d2
    }
    
    try:
        # Call API
        res = requests.post(f"{API_URL}/predict", json=payload, timeout=5)
        
        if res.status_code == 200:
            data = res.json()
            is_fraud = data["prediction"]
            prob = data["probability"]
            risk_level = data["risk_level"]
            lbl = data["label"]
            thresh = data["optimal_threshold"]
            
            st.divider()
            st.subheader("📊 Analysis Report Summary")
            
            res_col1, res_col2, res_col3 = st.columns(3)
            
            with res_col1:
                if is_fraud == 1:
                    st.error(f"Prediction: {lbl}")
                else:
                    st.success(f"Prediction: {lbl}")
                    
            with res_col2:
                st.metric("Fraud Probability", f"{prob * 100:.2f}%")
                
            with res_col3:
                if risk_level == "High":
                    st.error(f"Risk Profile: {risk_level}")
                elif risk_level == "Medium":
                    st.warning(f"Risk Profile: {risk_level}")
                else:
                    st.info(f"Risk Profile: {risk_level}")
                    
            st.info(f"💡 Decision boundary criteria: Probability >= {thresh:.4f} is flagged as Fraud.")
            
        else:
            st.error(f"API Error (HTTP {res.status_code}): {res.text}")
            
    except requests.exceptions.ConnectionError:
        st.error("❌ Connection failed! Could not reach Uvicorn API serving server on port 8000. Start the server first using: launch_project.sh.")
    except Exception as e:
        st.error(f"❌ Error during evaluation: {str(e)}")

# Server Diagnostics
st.divider()
st.subheader("🩺 Local Serving Status")
try:
    health_res = requests.get(f"{API_URL}/health", timeout=3)
    if health_res.status_code == 200:
        health_data = health_res.json()
        st.success(f"✓ API Server Status: HEALTHY | Model Features: {health_data.get('features_count')} | Model AUC-ROC: {health_data.get('auc_roc')}")
    else:
        st.warning(f"⚠️ API Server Online but Degraded (HTTP {health_res.status_code})")
except:
    st.error("❌ API Server Offline (FastAPI is not running on port 8000)")

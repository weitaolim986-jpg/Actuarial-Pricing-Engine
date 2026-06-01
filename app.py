import streamlit as st
import pandas as pd
import xgboost as xgb
import numpy as np
import pickle
import shap
import matplotlib.pyplot as plt

# --- 1. PAGE CONFIGURATION ---
st.set_page_config(page_title="Actuarial Pricing Engine", page_icon="🚗", layout="centered")

# --- 2. LOAD THE MODELS ---
@st.cache_resource
def load_models():
    with open('xgb_freq_model.pkl', 'rb') as f:
        freq_model = pickle.load(f)
    with open('xgb_seve_model.pkl', 'rb') as f:
        seve_model = pickle.load(f)
    return freq_model, seve_model

freq_model, seve_model = load_models()

# --- 3. FRONT-END UI ---
st.title("🚗 Actuarial ML Pricing Engine")
st.markdown("""
Welcome to the interactive auto insurance rating engine. 
Adjust the driver parameters below to see how our XGBoost model calculates the expected Pure Premium in real-time.
""")

st.divider()

# Input Form
with st.container():
    st.subheader("Driver Profile")
    col1, col2 = st.columns(2)
    
    with col1:
        driv_age = st.slider("Driver Age", min_value=18, max_value=90, value=30)
        veh_power = st.slider("Vehicle Power (Category)", min_value=4, max_value=15, value=6)
        
    with col2:
        bonus_malus = st.slider("Bonus-Malus (NCD Score, lower is safer)", min_value=50, max_value=230, value=50)
        exposure = st.slider("Policy Term (Years)", min_value=0.1, max_value=1.0, value=1.0, step=0.1)

# --- 4. BACK-END PREDICTION LOGIC ---
if st.button("Calculate Premium", type="primary"):
    
    # Package inputs into a DataFrame
    input_data = pd.DataFrame({
        'VehPower': [veh_power],
        'DrivAge': [driv_age],
        'BonusMalus': [bonus_malus]
    })
    
    # Predict Severity (No offset needed) using 'seve' naming convention
    dpred_seve = xgb.DMatrix(input_data)
    pred_seve = seve_model.predict(dpred_seve)[0]
    
    # Predict Frequency (Must include the Exposure offset!)
    dpred_freq = xgb.DMatrix(input_data)
    dpred_freq.set_base_margin(np.log([exposure]))
    pred_freq = freq_model.predict(dpred_freq)[0]
    
    # Calculate Pure Premium (Freq * Seve)
    pure_premium = pred_freq * pred_seve
    
    # --- 5. DISPLAY RESULTS ---
    st.divider()
    st.success(f"### Final Expected Pure Premium: ${pure_premium:,.2f}")
    
    res_col1, res_col2 = st.columns(2)
    res_col1.metric("Expected Claim Frequency", f"{pred_freq:.4f} claims/term")
    res_col2.metric("Expected Claim Severity", f"${pred_seve:,.2f} per claim")
    
    st.caption("Note: This is the technical pure premium. Commercial pricing would include expense loads, profit margins, and capping/collaring.")

    # --- 6. EXPLAINABLE AI (SHAP) ---
    st.divider()
    st.subheader("📊 Explainable AI: Claim Severity Driver Analysis")
    st.markdown("This SHAP Waterfall chart breaks down exactly how this driver's specific features pushed the expected cost per claim up or down from the portfolio average.")
    
    # Initialize the explainer and calculate SHAP values using seve_model
    explainer = shap.TreeExplainer(seve_model)
    shap_values = explainer(input_data)
    
    # Generate the plot
    fig = plt.figure(figsize=(8, 4))
    shap.plots.waterfall(shap_values[0], show=False)
    st.pyplot(plt.gcf())
    
    # Clear the figure from memory to prevent overlapping charts
    plt.clf()
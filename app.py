import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap

# === Load model, scaler, and feature names ===
model = joblib.load("xgboost_model.pkl")
scaler = joblib.load("scaler.pkl")
feature_names = joblib.load("feature_names.pkl")

st.set_page_config(
    page_title="Mutation Pathogenicity Predictor",
    layout="wide",  
    initial_sidebar_state="auto"
)

st.title("Mutation Pathogenicity Predictor")
st.markdown("Enter values for the **top 15 most important features**. Click **Predict** to see the result.\n")

with st.form("mutation_form"):
    func_nonsyn = st.selectbox("func_nonsynonymous SNV", ["Yes", "No"])
    phyloP = st.number_input("phyloP100way_vertebrate", step=0.000000001, format="%.9f")
    af_ami = st.number_input("AF_ami", step=0.000000001, format="%.9f")
    aromatic = st.number_input("aromatic", step=0.000000001, format="%.9f")
    af_male = st.number_input("AF_male", step=0.000000001, format="%.9f")
    func_stopgain = st.selectbox("func_stopgain", ["Yes", "No"])
    omim_ar = st.number_input("omim_Autosomal_recessive", step=0.000000001, format="%.9f")
    af_eas = st.number_input("AF_eas", step=0.000000001, format="%.9f")
    omim_other = st.number_input("omim_other", step=0.000000001, format="%.9f")
    af_amr = st.number_input("AF_amr", step=0.000000001, format="%.9f")
    phastCons = st.number_input("phastCons100way_vertebrate", step=0.000000001, format="%.9f")
    func_nonframeshift = st.selectbox("func_nonframeshift", ["Yes", "No"])
    siphy = st.number_input("SiPhy_29way_logOdds", step=0.000000001, format="%.9f")
    lof_score = st.number_input("lof_score", step=0.000000001, format="%.9f")
    blosum100 = st.number_input("blosum100", step=0.000000001, format="%.9f")

    submit = st.form_submit_button("Predict")

if submit:
    input_dict = {
        'func_nonsynonymous SNV': 1.0 if func_nonsyn == "Yes" else 0.0,
        'phyloP100way_vertebrate': phyloP,
        'AF_ami': af_ami,
        'aromatic': aromatic,
        'AF_male': af_male,
        'func_stopgain': 1.0 if func_stopgain == "Yes" else 0.0,
        'omim_Autosomal_recessive': omim_ar,
        'AF_eas': af_eas,
        'omim_other': omim_other,
        'AF_amr': af_amr,
        'phastCons100way_vertebrate': phastCons,
        'func_nonframeshift': 1.0 if func_nonframeshift == "Yes" else 0.0,
        'SiPhy_29way_logOdds': siphy,
        'lof_score': lof_score,
        'blosum100': blosum100,
    }

    full_input = {f: input_dict.get(f, 0.0) for f in feature_names}
    input_df = pd.DataFrame([full_input])
    X_scaled = pd.DataFrame(scaler.transform(input_df), columns=input_df.columns)

    # Prediction
    prediction = model.predict(X_scaled)[0]
    proba = model.predict_proba(X_scaled)[0]
    label = "Pathogenic" if prediction == 1 else "Benign"

    st.subheader(f"**Result: {label}**")
    st.caption(f"Confidence: {proba[prediction]*100:.1f}%")
    st.write(f"Probabilities → Benign: {proba[0]*100:.1f}%, Pathogenic: {proba[1]*100:.1f}%")

    # SHAP Text-Based Explanation
    st.markdown("---")
    st.subheader("Top Feature Contributions")
    st.caption("High Impact: |SHAP| > 0.1, Medium Impact: 0.05 < |SHAP| ≤ 0.1, Low Impact: |SHAP| ≤ 0.05")

    # Calculate SHAP values
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_scaled)
    sample_shap = shap_values[0]

    # Only keep SHAP values for features the user controlled
    user_features = list(input_dict.keys())
    impact_data = []

    for feature in user_features:
        shap_val = sample_shap[X_scaled.columns.get_loc(feature)]
        raw_val = input_df.iloc[0][feature]  # unscaled user input
        impact_data.append((feature, shap_val, raw_val))

    # Sort by absolute SHAP value
    impact_data.sort(key=lambda x: abs(x[1]), reverse=True)

    # Display all 15
    for i, (feature, shap_val, feature_val) in enumerate(impact_data, 1):
        abs_val = abs(shap_val)
        if abs_val > 0.1:
            impact = "High"
        elif abs_val > 0.05:
            impact = "Medium"
        else:
            impact = "Low"

        direction = "↑ Pathogenic" if shap_val > 0 else "↓ Benign"

        st.write(f"{i}. {feature} = {feature_val:.9f} → SHAP: {shap_val:.9f} ({impact} impact, {direction})")


import streamlit as st
import pandas as pd
import joblib

# Load model and preprocessing
model = joblib.load("model.pkl")
imputer = joblib.load("imputer.pkl")
scaler = joblib.load("scaler.pkl")
features = joblib.load("features.pkl")

st.set_page_config(page_title="Metro Station Civil Cost Predictor", layout="wide")
st.title("🚇 Metro Station Civil Cost Predictor (ML-Based)")
st.markdown("Predict estimated civil cost for metro stations using your trained machine learning model.")

# Sidebar: Autofill selector
st.sidebar.header("📊 Input Station Parameters")
station_type = st.sidebar.selectbox("Autofill with Station Type:", ["Custom Input", "Regular", "Terminal", "Interchange"])

# Define autofill presets
autofill_presets = {
    "Regular": [1] * len(features),
    "Terminal": [2] * len(features),
    "Interchange": [3] * len(features),
    "Custom Input": [0] * len(features)
}

# Build input form
user_inputs = {}
for i, feature in enumerate(features):
    user_inputs[feature] = st.sidebar.number_input(
        label=feature,
        value=autofill_presets[station_type][i],
        step=1.0
    )

user_df = pd.DataFrame([user_inputs])

# Predict single input
if st.button("🔮 Predict Civil Cost"):
    try:
        imputed = imputer.transform(user_df)
        scaled = scaler.transform(imputed)
        predicted_cost = model.predict(scaled)[0]
        st.success(f"💰 Predicted Civil Cost: ₹{predicted_cost:,.2f} Cr")
    except Exception as e:
        st.error(f"Error: {e}")

# Upload Excel file
st.subheader("📁 Upload Excel File for Batch Prediction")
uploaded_file = st.file_uploader("Upload Excel file (.xlsx)", type=["xlsx"])

if uploaded_file:
    try:
        df_uploaded = pd.read_excel(uploaded_file)
        missing = set(features) - set(df_uploaded.columns)
        if missing:
            st.error(f"Missing columns in uploaded file: {missing}")
        else:
            imputed = imputer.transform(df_uploaded[features])
            scaled = scaler.transform(imputed)
            predictions = model.predict(scaled)
            df_uploaded["Predicted Civil Cost (Cr)"] = predictions
            st.success("✅ Batch prediction completed.")
            st.dataframe(df_uploaded)
    except Exception as e:
        st.error(f"Error processing uploaded file: {e}")

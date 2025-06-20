
import streamlit as st
import pandas as pd
import joblib

# Load model and preprocessing artifacts
model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")
imputer = joblib.load("imputer.pkl")
features = joblib.load("features.pkl")

# Set page layout and title
st.set_page_config(page_title="Metro Station Civil Cost Estimator", layout="wide")
st.title("🚇 Metro Station Civil Cost Estimator")
st.markdown("Estimate metro station civil construction cost using 10 key design parameters.")

# Sidebar: Station Type Selection
st.sidebar.header("📑 Input Parameters")
station_type = st.sidebar.selectbox("Select Station Type:", ["Custom Input", "Regular", "Terminal", "Interchange"])

# Define autofill presets for each station type
autofill_presets = {
    "Regular": [1, 1, 3, 1.05, 160, 18, 12, 8, 2, 1],
    "Terminal": [2, 2, 4, 1.1, 180, 20, 15, 10, 3, 2],
    "Interchange": [3, 3, 5, 1.2, 200, 22, 18, 12, 4, 3],
    "Custom Input": [0] * len(features)
}

# Create user input form
user_inputs = {}
for i, feature in enumerate(features):
    default_value = autofill_presets[station_type][i]
    user_inputs[feature] = st.sidebar.number_input(
        label=feature,
        value=float(default_value),
        step=1.0
    )

# Prediction Button
if st.button("💰 Predict Civil Cost"):
    input_df = pd.DataFrame([user_inputs])
    input_imputed = imputer.transform(input_df)
    input_scaled = scaler.transform(input_imputed)
    prediction = model.predict(input_scaled)[0]
    st.success(f"🏗️ Estimated Civil Construction Cost: ₹ {prediction:,.2f} Cr")

# Batch Prediction Upload
st.markdown("### 📂 Batch Prediction via Excel")
uploaded_file = st.file_uploader("Upload Excel file with station parameters", type=["xlsx"])
if uploaded_file:
    batch_df = pd.read_excel(uploaded_file)
    try:
        batch_imputed = imputer.transform(batch_df)
        batch_scaled = scaler.transform(batch_imputed)
        batch_predictions = model.predict(batch_scaled)
        batch_df["Predicted Civil Cost (Cr)"] = batch_predictions
        st.dataframe(batch_df)
    except Exception as e:
        st.error(f"Batch prediction failed: {str(e)}")

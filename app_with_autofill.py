
import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="Metro Station Cost Predictor", layout="wide")

# Load models and metadata
model = joblib.load("model.pkl")
imputer = joblib.load("imputer.pkl")
scaler = joblib.load("scaler.pkl")
features = joblib.load("features.pkl")

st.title("🚇 Metro Station Civil Cost Predictor (ML-Based)")
st.markdown("Predict estimated civil cost for metro stations using your trained machine learning model.")

# Sidebar Autofill
st.sidebar.header("🧮 Input Station Parameters")
station_type = st.sidebar.selectbox("Autofill with Station Type:", ["Custom Input", "Regular", "Terminal", "Interchange"])

autofill_data = {
    "Regular": {f: 1 for f in features},
    "Terminal": {f: 2 for f in features},
    "Interchange": {f: 3 for f in features},
    "Custom Input": {f: 0 for f in features}
}
user_input = {f: st.sidebar.number_input(f, value=autofill_data[station_type][f]) for f in features}

if st.button("🔮 Predict Civil Cost"):
    try:
        df = pd.DataFrame([user_input])
        df_imputed = imputer.transform(df)
        df_scaled = scaler.transform(df_imputed)
        prediction = model.predict(df_scaled)[0]
        st.success(f"💰 Predicted Civil Cost: ₹{prediction:,.2f} Cr")
    except Exception as e:
        st.error(f"Error: {e}")

# Batch Upload
st.subheader("📄 Upload Excel File for Batch Prediction")
uploaded_file = st.file_uploader("Upload an Excel file with station data", type=["xlsx"])

if uploaded_file:
    try:
        batch_df = pd.read_excel(uploaded_file)
        missing = set(features) - set(batch_df.columns)
        if missing:
            st.error(f"Missing required columns: {missing}")
        else:
            imputed = imputer.transform(batch_df[features])
            scaled = scaler.transform(imputed)
            predictions = model.predict(scaled)
            result = batch_df.copy()
            result["Predicted Civil Cost (Cr)"] = predictions
            st.success("✅ Prediction complete!")
            st.dataframe(result)
    except Exception as e:
        st.error(f"Error during batch prediction: {e}")

import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load trained model
model = joblib.load("metro_cost_model.pkl")

st.set_page_config(page_title="Metro Cost Estimator", layout="wide")
st.title("🚇 Metro Civil Cost Estimator (ML-Based)")
st.markdown("Enter metro station design & site parameters to estimate civil construction cost (₹ crore).")

# --- Input Form ---
with st.form("cost_form"):
    col1, col2, col3 = st.columns(3)

    with col1:
        city = st.selectbox("City", ['Delhi', 'Mumbai', 'Kolkata', 'Bengaluru', 'Chennai', 'Hyderabad', 'Ahmedabad',
                                     'Pune', 'Jaipur', 'Lucknow', 'Kochi', 'Noida', 'Kanpur', 'Agra', 'Indore',
                                     'Navi Mumbai', 'Gurugram'])
        metro_type = st.selectbox("Metro Type", ['underground', 'elevated'])
        soil_type = st.selectbox("Soil Type", ['Alluvial', 'Rock', 'Clay', 'Mixed'])
        seismic_zone = st.selectbox("Seismic Zone", ['II', 'III', 'IV', 'V'])
        weather = st.selectbox("Weather Impact", ['Dry', 'Rainy', 'Flood Prone'])

    with col2:
        station_length = st.slider("Station Length (m)", 120, 250, 180)
        station_depth = st.slider("Station Depth (m)", 10, 30, 20)
        station_width = st.slider("Station Width (m)", 15, 30, 20)
        levels = st.selectbox("Number of Levels", [2, 3])
        typology = st.selectbox("Station Typology", ['Regular', 'Interchange', 'Terminal'])

    with col3:
        rcc_volume = st.slider("RCC Volume (cum)", 3000, 20000, 12000)
        labor_count = st.slider("Peak Labor Count", 200, 700, 450)
        regional_index = st.slider("Regional Cost Index", 0.85, 1.15, 1.00)
        diaphragm_wall_area = st.number_input("Diaphragm Wall Area (sqm)", min_value=0)
        tbm_used = st.selectbox("TBM Used", ['Yes', 'No'])
        tower_crane = st.selectbox("Tower Crane Required", ['Yes', 'No'])

    submitted = st.form_submit_button("🔮 Predict Cost")

# --- Prediction ---
if submitted:
    input_dict = {
        'City': city,
        'Metro_Type': metro_type,
        'Station_Depth_m': station_depth,
        'Station_Length_m': station_length,
        'Station_Width_m': station_width,
        'Levels': levels,
        'Station_Typology': typology,
        'Excavation_Volume_cum': round(station_length * station_width * station_depth * 1.15),
        'Diaphragm_Wall_Area_sqm': diaphragm_wall_area,
        'RCC_Volume_cum': rcc_volume,
        'Soil_Type': soil_type,
        'Seismic_Zone': seismic_zone,
        'Regional_Cost_Index': regional_index,
        'Weather_Impact': weather,
        'Peak_Labor_Count': labor_count,
        'TBM_Used': tbm_used,
        'Tower_Crane_Required': tower_crane,
        'Waterproofing_Area_sqm': station_length * station_depth * 0.25 if metro_type == "underground" else station_length * 0.25,
        'Foundation_Quantity': int(station_length * 1.1),
        'Shuttering_Area_sqm': round(rcc_volume * 1.4),
        'Reinforcement_TMT_tons': round(rcc_volume * 0.11, 2),
        'Structural_Steel_tons': 120 if metro_type == "underground" else 60,
        'Water_Table_m': 6,
        'Flood_Risk': 'No',
        'Heritage_Nearby': 'No',
        'Material_Inflation_Rate_percent': 6.0,
        'Exchange_Rate_Sensitivity': 'Medium',
        'Contract_Type': 'Item Rate',
        'Construction_Method': 'Top-down' if metro_type == 'underground' else 'Bottom-up',
        'Cement_Qty_tons': int(rcc_volume * 0.4),
        'Sand_Qty_cum': int(rcc_volume * 0.65),
        'Aggregate_Qty_cum': int(rcc_volume * 1.0),
        'Concrete_Grade': 'M30',
        'Waterproofing_Type': 'membrane',
        'Rebar_Type': 'Fe500D',
        'TBM_Diameter_m': 6.5 if tbm_used == "Yes" else 0,
        'Excavator_Count': 4,
        'Transit_Mixer_Day': 6,
        'Dewatering_Method': 'Wellpoint',
        'Gantry_DG_Setup': 'Yes'
    }

    input_df = pd.DataFrame([input_dict])
    cost_pred = model.predict(input_df)[0]
    st.success(f"💰 **Estimated Civil Cost: ₹ {round(cost_pred, 2)} crore**")

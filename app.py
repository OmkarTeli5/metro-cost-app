import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load model, encoder, and column list
model = joblib.load("metro_cost_model.pkl")
encoder = joblib.load("encoder.pkl")
column_order = joblib.load("model_columns.pkl")

st.set_page_config(page_title="Metro Cost Estimator", layout="wide")
st.title("🚇 Metro Civil Cost Estimator (ML-Based)")
st.markdown("Enter metro station design & site parameters to estimate civil construction cost (₹ crore).")

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

if submitted:
    input_dict = {
        'City': city,
        'Metro_Type': metro_type,
        'Station_Typology': typology,
        'Soil_Type': soil_type,
        'Seismic_Zone': seismic_zone,
        'Weather_Impact': weather,
        'TBM_Used': tbm_used,
        'Tower_Crane_Required': tower_crane,
        'Excavator_Count': 4,
        'Dewatering_Method': 'Wellpoint',
        'Gantry_DG_Setup': 'Yes',
        'Contract_Type': 'Item Rate',
        'Construction_Method': 'Top-down' if metro_type == 'underground' else 'Bottom-up',
        'Concrete_Grade': 'M30',
        'Waterproofing_Type': 'membrane',
        'Rebar_Type': 'Fe500D',
        'Exchange_Rate_Sensitivity': 'Medium',
        'Flood_Risk': 'No',
        'Heritage_Nearby': 'No',
        'Levels': levels
    }

    input_dict.update({
        'Station_Depth_m': station_depth,
        'Station_Length_m': station_length,
        'Station_Width_m': station_width,
        'Excavation_Volume_cum': round(station_length * station_width * station_depth * 1.15),
        'Diaphragm_Wall_Area_sqm': diaphragm_wall_area,
        'Waterproofing_Area_sqm': station_length * station_depth * 0.25 if metro_type == "underground" else station_length * 0.25,
        'RCC_Volume_cum': rcc_volume,
        'Shuttering_Area_sqm': round(rcc_volume * 1.4),
        'Reinforcement_TMT_tons': round(rcc_volume * 0.11, 2),
        'Structural_Steel_tons': 120 if metro_type == "underground" else 60,
        'Water_Table_m': 6,
        'Material_Inflation_Rate_percent': 6.0,
        'Regional_Cost_Index': regional_index,
        'Peak_Labor_Count': labor_count,
        'Foundation_Quantity': int(station_length * 1.1),
        'Cement_Qty_tons': int(rcc_volume * 0.4),
        'Sand_Qty_cum': int(rcc_volume * 0.65),
        'Aggregate_Qty_cum': int(rcc_volume * 1.0),
        'TBM_Diameter_m': 6.5 if tbm_used == "Yes" else 0
    })

    input_df = pd.DataFrame([input_dict])
    cat_cols = encoder.feature_names_in_
    encoded_input = pd.DataFrame(encoder.transform(input_df[cat_cols]),
                                 columns=encoder.get_feature_names_out(cat_cols))
    numeric_input = input_df.drop(columns=cat_cols).reset_index(drop=True)
    final_input = pd.concat([encoded_input, numeric_input], axis=1)

    # ✅ Final fix: reorder columns to match training
    final_input = final_input[column_order]

    cost = model.predict(final_input)[0]
    st.success(f"💰 Estimated Civil Cost: ₹ {round(cost, 2)} crore")

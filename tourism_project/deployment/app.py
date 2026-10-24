import streamlit as st
import pandas as pd
import joblib
import os

# Page configuration
st.set_page_config(
    page_title="Wellness Tourism Package Predictor", 
    page_icon="🌿", 
    layout="centered"
)

@st.cache_resource
def load_model():
    model_path = os.path.join(os.path.dirname(__file__), "model.pkl")
    return joblib.load(model_path)

model = load_model()

st.title("🌿 Wellness Tourism Package Purchase Predictor")
st.write("Enter customer details below to predict if they will purchase the Wellness Tourism Package.")

# Input fields organized cleanly
col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Age", min_value=18, max_value=100, value=35)
    monthly_income = st.number_input("Monthly Income", min_value=1000, max_value=200000, value=25000)
    city_tier = st.selectbox("City Tier", [1, 2, 3])
    passport = st.selectbox("Has Passport?", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")

with col2:
    own_car = st.selectbox("Owns Car?", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
    duration_of_pitch = st.number_input("Duration of Pitch (minutes)", min_value=1, max_value=120, value=15)
    number_of_followups = st.number_input("Number of Followups", min_value=0, max_value=10, value=3)
    pitch_satisfaction = st.slider("Pitch Satisfaction Score", 1, 5, 3)

if st.button("Predict Purchase Likelihood", type="primary"):
    # Build a dataframe matching the features expected by the model
    input_data = pd.DataFrame({
        'Age': [age],
        'MonthlyIncome': [monthly_income],
        'CityTier': [city_tier],
        'Passport': [passport],
        'OwnCar': [own_car],
        'DurationOfPitch': [duration_of_pitch],
        'NumberOfFollowups': [number_of_followups],
        'PitchSatisfactionScore': [pitch_satisfaction]
    })
    
    try:
        prediction = model.predict(input_data)[0]
        if prediction == 1:
            st.success("🎉 **High Likelihood:** The customer is likely to purchase the Wellness Tourism Package!")
        else:
            st.info("💡 **Low Likelihood:** The customer is unlikely to purchase the package at this time.")
    except Exception as e:
        # Fallback if feature columns need dummy alignment
        st.error(f"Prediction note: Ensure all categorical columns match training features. Error: {e}")

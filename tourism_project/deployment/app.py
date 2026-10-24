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

# Input fields organized into clean tabs or sections
with st.form("prediction_form"):
    st.subheader("Customer Demographics & Profile")
    col1, col2 = st.columns(2)

    with col1:
        age = st.number_input("Age", min_value=18, max_value=100, value=35)
        monthly_income = st.number_input("Monthly Income", min_value=1000, max_value=200000, value=25000)
        city_tier = st.selectbox("City Tier", [1, 2, 3])
        occupation = st.selectbox("Occupation", ["Salaried", "Small Business", "Large Business", "Free Lancer"])
        gender = st.selectbox("Gender", ["Male", "Female"])
        marital_status = st.selectbox("Marital Status", ["Single", "Married", "Unmarried", "Divorced"])

    with col2:
        designation = st.selectbox("Designation", ["Executive", "Manager", "Senior Manager", "AVP", "VP"])
        passport = st.selectbox("Has Passport?", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
        own_car = st.selectbox("Owns Car?", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
        number_of_children = st.number_input("Number of Children Visiting", min_value=0, max_value=5, value=1)
        number_of_persons = st.number_input("Number of Persons Visiting", min_value=1, max_value=10, value=2)

    st.subheader("Trip & Pitch Details")
    col3, col4 = st.columns(2)

    with col3:
        type_of_contact = st.selectbox("Type of Contact", ["Self Enquiry", "Company Invited"])
        duration_of_pitch = st.number_input("Duration of Pitch (minutes)", min_value=1, max_value=120, value=15)
        number_of_followups = st.number_input("Number of Followups", min_value=0, max_value=10, value=3)

    with col4:
        product_pitched = st.selectbox("Product Pitched", ["Basic", "Standard", "Deluxe", "Super Deluxe", "King"])
        preferred_property_star = st.slider("Preferred Property Star", 3.0, 5.0, 3.0, 1.0)
        number_of_trips = st.number_input("Number of Trips per Year", min_value=1, max_value=20, value=2)
        pitch_satisfaction = st.slider("Pitch Satisfaction Score", 1, 5, 3)

    submit_button = st.form_submit_button("Predict Purchase Likelihood", type="primary")

if submit_button:
    # 1. Build a raw dataframe with user inputs
    input_data = pd.DataFrame({
        'Unnamed: 0': [0], # Dummy placeholder for Unnamed: 0 if in training columns
        'Age': [age],
        'CityTier': [city_tier],
        'DurationOfPitch': [duration_of_pitch],
        'Occupation': [occupation],
        'Gender': [gender],
        'NumberOfPersonVisiting': [number_of_persons],
        'NumberOfFollowups': [number_of_followups],
        'ProductPitched': [product_pitched],
        'PreferredPropertyStar': [preferred_property_star],
        'MaritalStatus': [marital_status],
        'NumberOfTrips': [number_of_trips],
        'Passport': [passport],
        'PitchSatisfactionScore': [pitch_satisfaction],
        'OwnCar': [own_car],
        'NumberOfChildrenVisiting': [number_of_children],
        'Designation': [designation],
        'MonthlyIncome': [monthly_income],
        'TypeofContact': [type_of_contact]
    })
    
    try:
        # 2. Apply get_dummies matching the training preprocessing
        input_encoded = pd.get_dummies(input_data, drop_first=True)
        
        # 3. Retrieve model feature names if available, or align columns directly
        if hasattr(model, "feature_names_in_"):
            input_encoded = input_encoded.reindex(columns=model.feature_names_in_, fill_value=0)
        
        # 4. Make prediction
        prediction = model.predict(input_encoded)[0]
        
        if prediction == 1:
            st.success("🎉 **High Likelihood:** The customer is likely to purchase the Wellness Tourism Package!")
        else:
            st.info("💡 **Low Likelihood:** The customer is unlikely to purchase the package at this time.")
            
    except Exception as e:
        st.error(f"Prediction error: {e}")

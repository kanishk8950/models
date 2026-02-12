import streamlit as st
import pandas as pd
import joblib
import requests
import numpy as np

st.set_page_config(
    page_title="Alzheimer's Disease Prediction",
    page_icon="🧠",
    layout="centered",
    menu_items={
        'Get Help': None,
        'Report a bug': None,
        'About': None
    }
)

# Hide deploy button and menu
hide_menu_style = """
    <style>
    .stBaseButton-header {display:none;}
    [data-testid="stBaseButton-header"] {display:none !important;}
    </style>
"""
st.markdown(hide_menu_style, unsafe_allow_html=True)

# --- Load model and encoders ---
@st.cache_resource
def load_models():
    model = joblib.load("alzheimers_model.pkl")
    label_encoders = joblib.load("label_encoders.pkl")
    target_encoder = joblib.load("target_encoder.pkl")
    model_features = joblib.load("model_features.pkl")
    return model, label_encoders, target_encoder, model_features

try:
    model, label_encoders, target_encoder, model_features = load_models()
    st.success("✅ Model loaded successfully!")
except Exception as e:
    st.error(f"❌ Failed to load model: {str(e)}")
    st.stop()

st.title("🧠 Alzheimer's Disease Risk Prediction")
st.write("Provide patient details to assess Alzheimer's disease risk based on clinical data.")

# --- Get query params - Compatible with older Streamlit versions ---
try:
    # Try the new method first (Streamlit 1.29.0+)
    query_params = st.query_params
    API_ENDPOINT = query_params.get("url", "")
    access_token = query_params.get("accessToken", "")
except AttributeError:
    # Fallback to older method (Streamlit < 1.29.0)
    try:
        query_params = st.experimental_get_query_params()
        API_ENDPOINT = query_params.get("url", [""])[0]
        access_token = query_params.get("accessToken", [""])[0]
    except AttributeError:
        # If both fail, just set empty values
        API_ENDPOINT = ""
        access_token = ""

# --- Helper to safely fetch encoder classes ---
def get_encoder_classes(column_name, default=["0", "1"]):
    """Return label encoder classes or fallback."""
    if column_name in label_encoders:
        return label_encoders[column_name].classes_
    else:
        return default

# --- Collect user input based on actual dataset features ---
def user_input():
    data = {}

    st.subheader("📋 Demographics & Physical Measurements")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        data["Age"] = st.slider("Age", 60, 90, 75, help="Patient's age in years")
        data["Gender"] = st.selectbox("Gender", options=[0, 1], format_func=lambda x: "Male" if x == 1 else "Female")
        data["EducationLevel"] = st.selectbox("Education Level", options=[0, 1, 2, 3], 
                                             format_func=lambda x: ["None", "Primary", "Secondary", "Higher"][x])
    
    with col2:
        data["BMI"] = st.slider("BMI", 15.0, 40.0, 25.0, help="Body Mass Index")
        data["Smoking"] = st.selectbox("Smoking Status", options=[0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
        data["AlcoholConsumption"] = st.slider("Alcohol Consumption", 0.0, 20.0, 5.0, help="Weekly alcohol consumption")
    
    with col3:
        data["PhysicalActivity"] = st.slider("Physical Activity", 0.0, 10.0, 5.0, help="Hours per week")
        data["DietQuality"] = st.slider("Diet Quality", 0.0, 10.0, 5.0, help="Diet quality score")
        data["SleepQuality"] = st.slider("Sleep Quality", 0.0, 10.0, 6.0, help="Sleep quality score")

    st.subheader("🏥 Medical History & Conditions")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        data["FamilyHistoryAlzheimers"] = st.selectbox("Family History of Alzheimer's", options=[0, 1], 
                                                       format_func=lambda x: "Yes" if x == 1 else "No")
        data["CardiovascularDisease"] = st.selectbox("Cardiovascular Disease", options=[0, 1], 
                                                     format_func=lambda x: "Yes" if x == 1 else "No")
        data["Diabetes"] = st.selectbox("Diabetes", options=[0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
    
    with col2:
        data["Depression"] = st.selectbox("Depression", options=[0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
        data["HeadInjury"] = st.selectbox("Head Injury History", options=[0, 1], 
                                          format_func=lambda x: "Yes" if x == 1 else "No")
        data["Hypertension"] = st.selectbox("Hypertension", options=[0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
    
    with col3:
        data["SystolicBP"] = st.slider("Systolic BP", 90, 180, 120, help="Systolic blood pressure")
        data["DiastolicBP"] = st.slider("Diastolic BP", 60, 120, 80, help="Diastolic blood pressure")

    st.subheader("🩺 Clinical Measurements")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        data["CholesterolTotal"] = st.slider("Total Cholesterol", 150.0, 300.0, 220.0, help="mg/dL")
        data["CholesterolLDL"] = st.slider("LDL Cholesterol", 50.0, 200.0, 120.0, help="mg/dL")
    
    with col2:
        data["CholesterolHDL"] = st.slider("HDL Cholesterol", 20.0, 100.0, 50.0, help="mg/dL")
        data["CholesterolTriglycerides"] = st.slider("Triglycerides", 50.0, 400.0, 150.0, help="mg/dL")
    
    with col3:
        data["MMSE"] = st.slider("MMSE Score", 0.0, 30.0, 25.0, help="Mini-Mental State Examination score")
        data["FunctionalAssessment"] = st.slider("Functional Assessment", 0.0, 10.0, 5.0, help="Functional assessment score")

    st.subheader("🧠 Cognitive & Behavioral Symptoms")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        data["MemoryComplaints"] = st.selectbox("Memory Complaints", options=[0, 1], 
                                                format_func=lambda x: "Yes" if x == 1 else "No")
        data["BehavioralProblems"] = st.selectbox("Behavioral Problems", options=[0, 1], 
                                                  format_func=lambda x: "Yes" if x == 1 else "No")
    
    with col2:
        data["ADL"] = st.selectbox("ADL Difficulties", options=[0, 1], 
                                   format_func=lambda x: "Yes" if x == 1 else "No", 
                                   help="Activities of Daily Living")
        data["Confusion"] = st.selectbox("Confusion", options=[0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
    
    with col3:
        data["Disorientation"] = st.selectbox("Disorientation", options=[0, 1], 
                                              format_func=lambda x: "Yes" if x == 1 else "No")
        data["PersonalityChanges"] = st.selectbox("Personality Changes", options=[0, 1], 
                                                  format_func=lambda x: "Yes" if x == 1 else "No")
    
    col1, col2 = st.columns(2)
    with col1:
        data["DifficultyCompletingTasks"] = st.selectbox("Difficulty Completing Tasks", options=[0, 1], 
                                                         format_func=lambda x: "Yes" if x == 1 else "No")
    with col2:
        data["Forgetfulness"] = st.selectbox("Forgetfulness", options=[0, 1], 
                                             format_func=lambda x: "Yes" if x == 1 else "No")

    return pd.DataFrame([data])

# --- Collect user inputs ---
input_df = user_input()
original_input_data = input_df.iloc[0].to_dict()

# --- Predict button ---
if st.button("🔍 Predict Alzheimer's Risk", use_container_width=True, type="primary"):
    try:
        encoded_df = input_df.copy()

        # Ensure all model features are present
        for feature in model_features:
            if feature not in encoded_df.columns:
                encoded_df[feature] = 0
        
        # Reorder columns to match model training
        encoded_df = encoded_df[model_features]

        # --- Predict ---
        proba = model.predict_proba(encoded_df)[0]
        prediction = model.predict(encoded_df)[0]
        
        # Get the label (Diagnosis: 0 or 1)
        label_idx = prediction
        label = target_encoder.inverse_transform([label_idx])[0]
        
        # Get confidence for positive class (Alzheimer's)
        confidence = proba[1] if len(proba) > 1 else proba[0]
        
        # --- Display results ---
        st.subheader("🧾 Prediction Results:")
        
        # Create columns for results
        result_col1, result_col2 = st.columns(2)
        
        with result_col1:
            if label == 1:
                st.error(f"### ⚠️ High Risk")
                st.metric("Prediction", "Alzheimer's Disease", delta="Positive", delta_color="inverse")
            else:
                st.success(f"### ✅ Low Risk")
                st.metric("Prediction", "No Alzheimer's", delta="Negative", delta_color="normal")
        
        with result_col2:
            st.metric("Confidence", f"{confidence*100:.1f}%")
            st.metric("MMSE Score", f"{original_input_data['MMSE']:.1f}", 
                     delta="Normal" if original_input_data['MMSE'] > 24 else "Concerning" if original_input_data['MMSE'] > 20 else "Severe")

        # Risk factors analysis
        st.subheader("📊 Risk Factor Analysis")
        
        risk_factors = []
        if original_input_data.get('FamilyHistoryAlzheimers', 0) == 1:
            risk_factors.append("• Family history of Alzheimer's")
        if original_input_data.get('Age', 0) > 80:
            risk_factors.append("• Age > 80 years")
        if original_input_data.get('CardiovascularDisease', 0) == 1:
            risk_factors.append("• Cardiovascular disease")
        if original_input_data.get('Diabetes', 0) == 1:
            risk_factors.append("• Diabetes")
        if original_input_data.get('Depression', 0) == 1:
            risk_factors.append("• Depression")
        if original_input_data.get('HeadInjury', 0) == 1:
            risk_factors.append("• History of head injury")
        if original_input_data.get('Hypertension', 0) == 1:
            risk_factors.append("• Hypertension")
        if original_input_data.get('Smoking', 0) == 1:
            risk_factors.append("• Smoking")
        if original_input_data.get('MMSE', 30) < 24:
            risk_factors.append(f"• Low MMSE score ({original_input_data['MMSE']:.1f})")
        
        if risk_factors:
            st.warning("### Identified Risk Factors:")
            for factor in risk_factors[:5]:  # Show top 5
                st.write(factor)
        else:
            st.info("No major risk factors identified.")

        with st.expander("📈 View Detailed Probabilities"):
            prob_df = pd.DataFrame({
                'Diagnosis': target_encoder.inverse_transform([0, 1]) if len(target_encoder.classes_) == 2 else target_encoder.classes_,
                'Probability': [f"{p*100:.2f}%" for p in proba]
            })
            st.dataframe(prob_df, use_container_width=True, hide_index=True)

        # --- Push to API ---
        if access_token and API_ENDPOINT:
            payload = {
                "model_type": "alzheimer_disease",
                "inputs": original_input_data,
                "prediction": "Positive" if label == 1 else "Negative",
                "confidence": float(confidence),
                "mmse_score": float(original_input_data.get('MMSE', 0)),
                "age": int(original_input_data.get('Age', 0))
            }
            headers = {
                "Authorization": f"Bearer {access_token}",
                "Content-Type": "application/json"
            }
            
            with st.spinner("📤 Uploading prediction to dashboard..."):
                try:
                    response = requests.post(API_ENDPOINT, json=payload, headers=headers, timeout=10)
                    if response.status_code in (200, 201):
                        st.success("✅ Prediction saved to your dashboard successfully!")
                    else:
                        st.warning(f"⚠️ API responded with status {response.status_code}")
                except requests.exceptions.Timeout:
                    st.error("❌ Request timed out. Please check your connection.")
                except requests.exceptions.ConnectionError:
                    st.error("❌ Failed to connect to the endpoint.")
                except Exception as e:
                    st.error(f"❌ Failed to upload prediction: {str(e)}")
        else:
            if not API_ENDPOINT and not access_token:
                st.info("ℹ️ No API credentials provided. Prediction not saved to dashboard.")

    except Exception as e:
        st.error(f"❌ Prediction error: {str(e)}")
        st.exception(e)  # This will show the full error traceback for debugging

# --- Feature Information ---
with st.expander("ℹ️ About the Model"):
    st.write("""
    ### Alzheimer's Disease Risk Assessment Model
    
    This model is trained on clinical data to assess the risk of Alzheimer's disease based on:
    
    **Key Risk Factors:**
    - Age (primary risk factor)
    - Family history of Alzheimer's
    - Cardiovascular health (hypertension, diabetes)
    - Cognitive assessment scores (MMSE)
    - Lifestyle factors (smoking, physical activity)
    - Medical history (head injury, depression)
    
    **Model Performance:**
    - Algorithm: Random Forest Classifier
    - Features used: 30+ clinical variables
    - Output: Probability of Alzheimer's disease
    
    **Important Note:** This tool is for research and educational purposes only. 
    It should not replace professional medical diagnosis. Always consult with healthcare 
    providers for proper evaluation and diagnosis.
    """)

# --- Footer ---
st.markdown("---")
st.caption("💡 This prediction is for informational purposes only. Please consult a healthcare professional for medical advice.")
st.caption("📊 Model trained on Alzheimer's Disease Dataset | For research use only")
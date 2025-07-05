import streamlit as st
import numpy as np
import pickle
import os

# 📦 Load the Trained Model
with open('heart_disease_model.pkl', 'rb') as file:
    model = pickle.load(file)

# 📏 Load custom threshold if available
default_threshold = 0.5
threshold_path = "threshold.txt"
if os.path.exists(threshold_path):
    with open(threshold_path, "r") as f:
        threshold = float(f.read().strip())
else:
    threshold = default_threshold

st.title("💓 Heart Disease Risk Prediction")

# 🌟 User Input Form
sex = st.selectbox("Sex", ["Female", "Male"])
gen_health = st.selectbox("General Health", ["Excellent", "Fair", "Good", "Poor", "Very good"])
age_category = st.selectbox("Age Category", [
    "18-24", "25-29", "30-34", "35-39", "40-44", "45-49",
    "50-54", "55-59", "60-64", "65-69", "70-74", "75-79", "80 or older"
])
smoking = st.selectbox("Do you smoke?", ["no", "yes"])
alcohol = st.selectbox("Do you drink alcohol?", ["No", "Yes"])
stroke = st.selectbox("History of stroke?", ["No", "Yes"])
diff_walking = st.selectbox("Difficulty walking?", ["No", "Yes"])
physical_activity = st.selectbox("Physical activity?", ["No", "Yes"])
asthma = st.selectbox("Do you have asthma?", ["No", "Yes"])
kidney_disease = st.selectbox("Do you have kidney disease?", ["No", "Yes"])
skin_cancer = st.selectbox("Do you have skin cancer?", ["No", "Yes"])
diabetic = st.selectbox("Diabetic Status", ["No", "No, borderline diabetes", "Yes", "Yes (during pregnancy)"])
race = st.selectbox("Race", [
    "American Indian/Alaskan Native", "Asian", "Black", "Hispanic", "Other", "White"
])

bmi = st.number_input("BMI", min_value=10.0, max_value=60.0, value=25.0)
physical_health = st.number_input("Physical Health (Days Unwell)", min_value=0, max_value=30, value=0)
mental_health = st.number_input("Mental Health (Days Unwell)", min_value=0, max_value=30, value=0)
sleep_time = st.number_input("Average Sleep Time (hours)", min_value=1, max_value=24, value=7)

# ✅ Mapping Dictionaries
sex_map = {"Female": 0, "Male": 1}
gen_health_map = {"Excellent": 4, "Fair": 1, "Good": 2, "Poor": 3, "Very good": 3}  # Fix if needed
age_map = {"18-24": 0, "25-29": 1, "30-34": 2, "35-39": 3, "40-44": 4, "45-49": 5,
           "50-54": 6, "55-59": 7, "60-64": 8, "65-69": 9, "70-74": 10, "75-79": 11, "80 or older": 12}
binary_map = {"No": 0, "Yes": 1, "no": 0, "yes": 1}
diabetic_map = {"No": 0, "No, borderline diabetes": 1, "Yes": 2, "Yes (during pregnancy)": 3}
race_map = {"American Indian/Alaskan Native": 0, "Asian": 1, "Black": 2, "Hispanic": 3, "Other": 4, "White": 5}

# 📥 Prepare Input Vector (17 Features)
input_data = np.array([[ 
    bmi,
    binary_map[smoking],
    binary_map[alcohol],
    binary_map[stroke],
    physical_health,
    mental_health,
    binary_map[diff_walking],
    sex_map[sex],
    age_map[age_category] * 0.5,
    race_map[race],
    diabetic_map[diabetic],
    binary_map[physical_activity],
    gen_health_map[gen_health],
    sleep_time,
    binary_map[asthma],
    binary_map[kidney_disease],
    binary_map[skin_cancer]
]])

# 🚀 Make Prediction
if st.button("Predict Heart Disease Risk"):
    prob = model.predict_proba(input_data)[0][1]
    prediction = int(prob >= threshold)
    confidence = prob * 100 if prediction == 1 else (1 - prob) * 100

    if prediction == 1:
        st.error(f"⚠️ High Risk of Heart Disease!\n\n🧠 Confidence: {confidence:.2f}%")
    else:
        st.success(f"✅ Low Risk of Heart Disease.\n\n🧠 Confidence: {confidence:.2f}%")

    # Optional debugging output
    st.markdown(f"🔍 **Predicted Probability (Risk):** `{prob:.2%}`")
    st.markdown(f"🎯 **Using Threshold:** `{threshold}`")

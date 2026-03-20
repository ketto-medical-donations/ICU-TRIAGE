import streamlit as st
import numpy as np
import joblib

# ----------- LOAD MODEL -----------

@st.cache_resource
def load_model():
    return joblib.load("vitals_rf_model.pkl")

model = load_model()

# ----------- PAGE CONFIG -----------

st.set_page_config(page_title="Patient Condition Predictor", layout="centered")

st.title("🏥 Patient Condition Classifier")
st.write("Enter patient vitals:")

# ----------- INPUT GRID (3 x 2) -----------

col1, col2 = st.columns(2)

with col1:
    with st.container(border=True):
        RR = st.number_input("Respiratory Rate (RR)", 0, 50, 18)
        Temp = st.number_input("Temperature (°C)", 25.0, 45.0, 37.0)
        SPO2 = st.number_input("SpO2 (%)", 0, 100, 98)

with col2:
    with st.container(border=True):
        HR = st.number_input("Heart Rate (HR)", 0, 200, 80)
        DBP = st.number_input("Diastolic BP (DBP)", 0, 150, 80)
        SBP = st.number_input("Systolic BP (SBP)", 0, 250, 120)

# ----------- NORMAL RANGES -----------

normal_ranges = {
    "HR": (55, 110),
    "RR": (12, 25),
    "SPO2": (93, 100),
    "Temp": (35, 38),
    "DBP": (60, 90),
    "SBP": (100, 140)
}

# ----------- Z-SCORE FUNCTION -----------

def compute_z_score(value, low, high):
    mean = (low + high) / 2
    std = (high - low) / 4  # approximate std
    return abs((value - mean) / std)

# ----------- PREDICT BUTTON -----------

st.markdown("---")

if st.button("🔍 Predict Condition"):

    input_data = np.array([[RR, HR, SBP, DBP, SPO2, Temp]])
    prediction = model.predict(input_data)[0]

    st.markdown("## 🧾 Result")

    # ----------- CONDITION OUTPUT -----------

    if prediction == "stable":
        st.success("🟢 Condition: STABLE")

    elif prediction == "unstable":
        st.warning("🟡 Condition: UNSTABLE")

    else:
        st.error("🔴 Condition: CRITICAL")

    # ----------- ANALYZE ABNORMAL VITALS -----------

    patient = {
        "HR": HR,
        "RR": RR,
        "SPO2": SPO2,
        "Temp": Temp,
        "DBP": DBP,
        "SBP": SBP
    }

    abnormal = []

    for vital, value in patient.items():
        low, high = normal_ranges[vital]

        if value < low or value > high:
            z = compute_z_score(value, low, high)
            abnormal.append((vital, value, z))

    # ----------- SORT BY SEVERITY -----------

    abnormal_sorted = sorted(abnormal, key=lambda x: x[2], reverse=True)

    # ----------- DISPLAY ABNORMALITIES -----------

    if abnormal_sorted:
        st.markdown("### ⚠️ Abnormal Vitals (Ranked by Severity)")

        for i, (vital, value, z) in enumerate(abnormal_sorted):
            if i == 0:
                st.error(f"🔴 MOST CRITICAL: {vital} = {value} ")
            else:
                st.warning(f"{vital} = {value}")
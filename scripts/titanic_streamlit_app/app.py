import streamlit as st
import numpy as np
import pandas as pd
import joblib
import tensorflow as tf
from tensorflow.keras.models import load_model

# -------------------------------
# SECTION 1 — Header Area
# -------------------------------
st.set_page_config(page_title="Titanic Survival Prediction", page_icon="🚢", layout="centered")

st.title("🚢 Titanic Survival Prediction System")
st.subheader("Deep Learning Based Passenger Survival Prediction")

# -------------------------------
# SECTION 2 — Project Description
# -------------------------------
st.markdown("""
This application predicts whether a passenger would survive during an emergency situation on the Titanic.  
It uses an **Artificial Neural Network (ANN)** built with **TensorFlow/Keras** and deployed via Streamlit.  
You can experiment with different passenger details and see survival probabilities in real-time.
""")

# -------------------------------
# Load Saved Scaler
# -------------------------------
@st.cache_resource
def load_scaler():
    return joblib.load("titanic_minmaxscaler.pkl")

scaler = load_scaler()


# -------------------------------
# SECTION 3 — Passenger Input Form
# -------------------------------
st.header("🧑 Passenger Input Form")

col1, col2, col3 = st.columns(3)

with col1:
    pclass = st.selectbox("Passenger Class", [1, 2, 3])

with col2:
    age = st.slider("Age", min_value=1, max_value=80, value=24)

with col3:
    fare = st.number_input("Fare", min_value=0.0, max_value=600.0, value=120.0)

# -------------------------------
# SECTION 5 — Model Selection
# -------------------------------
st.header("⚙️ Model Selection")
model_choice = st.radio("Choose Model:", ["titanic_model_0.keras", "titanic_model_1.keras"])

# Load selected model
@st.cache_resource
def load_selected_model(model_name):
    return load_model(model_name)

model = load_selected_model(model_choice)

# -------------------------------
# SECTION 4 — Prediction Button
# -------------------------------
if st.button("🔮 Predict Survival"):
    # Preprocess input
    input_data = np.array([[pclass, age, fare]])
    input_scaled = scaler.transform(input_data)

    # Prediction
    prediction = model.predict(input_scaled, verbose=0)
    prob = prediction[0][0]

    # -------------------------------
    # SECTION 5 — Prediction Output
    # -------------------------------
    st.success("✅ Prediction Complete!")
    if prob > 0.5:
        st.metric(label="Prediction Result", value="Survived")
    else:
        st.metric(label="Prediction Result", value="Not Survived")

    st.metric(label="Survival Probability", value=f"{prob:.2f}")
    confidence = prob if prob > 0.5 else (1 - prob)
    st.metric(label="Confidence Score", value=f"{confidence*100:.1f}%")

    # -------------------------------
    # SECTION 6 — Visualization Area
    # -------------------------------
    st.header("📊 Probability Visualization")

    survival_prob = prob
    non_survival_prob = 1 - prob

    chart_data = pd.DataFrame({
        "Status": ["Survived", "Not Survived"],
        "Probability": [survival_prob, non_survival_prob]
    })

    st.bar_chart(chart_data.set_index("Status"))

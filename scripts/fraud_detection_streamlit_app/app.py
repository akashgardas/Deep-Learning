import streamlit as st
import torch
import joblib
import pandas as pd
import numpy as np
from pathlib import Path

from model import FraudNet

# Current file
APP_DIR = Path(__file__).resolve().parent

# DEEP-LEARNING root
ROOT_DIR = APP_DIR.parent.parent

# Assets directory
ASSETS_DIR = ROOT_DIR / "models" / "fraud detection assets"

MODEL_PATH = ASSETS_DIR / "fraud_model.pth"
AMOUNT_SCALER_PATH = ASSETS_DIR / "amount_scaler.pkl"
TIME_SCALER_PATH = ASSETS_DIR / "time_scaler.pkl"
FEATURE_COLUMNS_PATH = ASSETS_DIR / "feature_columns.pkl"
THRESHOLD_PATH = ASSETS_DIR / "threshold.pkl"

# ----------------------------------
# PAGE CONFIG
# ----------------------------------

st.set_page_config(
    page_title="Fraud Detection System",
    page_icon="💳",
    layout="wide"
)

# ----------------------------------
# CUSTOM CSS
# ----------------------------------

st.markdown("""
<style>

.main {
    background-color: #0E1117;
}

.big-title {
    font-size:40px;
    font-weight:bold;
    color:white;
}

.metric-card {
    background-color:#1E222B;
    padding:20px;
    border-radius:15px;
    text-align:center;
}

.result-safe {
    background-color:#0F5132;
    padding:20px;
    border-radius:15px;
}

.result-fraud {
    background-color:#842029;
    padding:20px;
    border-radius:15px;
}

</style>
""", unsafe_allow_html=True)

# ----------------------------------
# LOAD ASSETS
# ----------------------------------

@st.cache_resource
def load_assets():

    model = FraudNet()

    model.load_state_dict(
        torch.load(
            MODEL_PATH,
            map_location="cpu"
        )
    )

    model.eval()

    assets = {
        "model": model,
        "amount_scaler": joblib.load(AMOUNT_SCALER_PATH),
        "time_scaler": joblib.load(TIME_SCALER_PATH),
        "feature_columns": joblib.load(FEATURE_COLUMNS_PATH),
        "threshold": joblib.load(THRESHOLD_PATH)
    }

    return assets

assets = load_assets()
model = assets["model"]
amount_scaler = assets["amount_scaler"]
time_scaler = assets["time_scaler"]
feature_columns = assets["feature_columns"]
threshold = assets["threshold"]

# ----------------------------------
# HEADER
# ----------------------------------

st.markdown(
    "<div class='big-title'>💳 Credit Card Fraud Detection System</div>",
    unsafe_allow_html=True
)

st.markdown(
    "Deep Learning powered transaction risk analysis"
)

st.divider()

# ----------------------------------
# TOP METRICS
# ----------------------------------

col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        "Model",
        "FraudNet"
    )

with col2:
    st.metric(
        "Threshold",
        f"{threshold:.2f}"
    )

with col3:
    st.metric(
        "Features",
        "30"
    )

st.divider()

# ----------------------------------
# INPUT AREA
# ----------------------------------

st.subheader("Transaction Information")

col1, col2 = st.columns(2)

with col1:

    amount = st.number_input(
        "Transaction Amount",
        min_value=0.0,
        value=100.0
    )

with col2:

    transaction_time = st.number_input(
        "Transaction Time",
        min_value=0.0,
        value=10000.0
    )

# ----------------------------------
# ADVANCED FEATURES
# ----------------------------------

with st.expander("Advanced PCA Features (V1-V28)"):

    feature_values = []

    cols = st.columns(4)

    for i in range(28):

        with cols[i % 4]:

            value = st.number_input(
                f"V{i+1}",
                value=0.0,
                key=f"v{i}"
            )

            feature_values.append(value)

# ----------------------------------
# PREDICTION BUTTON
# ----------------------------------

if st.button(
    "🔍 Analyze Transaction",
    use_container_width=True
):

    scaled_amount = amount_scaler.transform(
        [[amount]]
    )[0][0]

    scaled_time = time_scaler.transform(
        [[transaction_time]]
    )[0][0]

    input_data = [
        scaled_amount,
        scaled_time,
        *feature_values
    ]

    x = torch.tensor(
        [input_data],
        dtype=torch.float32
    )

    with torch.no_grad():

        output = model(x)

        probability = torch.sigmoid(
            output
        ).item()

    prediction = int(
        probability > threshold
    )

    st.divider()

    st.subheader("Analysis Result")

    st.progress(probability)

    st.metric(
        "Fraud Probability",
        f"{probability*100:.2f}%"
    )

    if prediction == 1:

        st.markdown(
            f"""
            <div class='result-fraud'>
            <h2>🚨 FRAUD DETECTED</h2>
            <p>High Risk Transaction</p>
            </div>
            """,
            unsafe_allow_html=True
        )

        st.error(
            "Recommendation: Block transaction and initiate verification."
        )

    else:

        st.markdown(
            f"""
            <div class='result-safe'>
            <h2>✅ LEGITIMATE TRANSACTION</h2>
            <p>Low Risk Transaction</p>
            </div>
            """,
            unsafe_allow_html=True
        )

        st.success(
            "Recommendation: Approve transaction."
        )
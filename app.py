import streamlit as st
import pandas as pd
import joblib

# =====================================================
# Page Config
# =====================================================
st.set_page_config(
    page_title="BODY Fraud Detection HUB",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# =====================================================
# Load Model
# =====================================================
model = joblib.load("model/model.pkl")
scaler = joblib.load("model/scaler.pkl")

FEATURES = [
    'Unnamed: 0',
    'cc_num',
    'amt',
    'zip',
    'lat',
    'long',
    'city_pop',
    'unix_time',
    'merch_lat',
    'merch_long'
]

THRESHOLD = 0.01

# =====================================================
# DARK FINTECH THEME (MATCHING EXAMPLE)
# =====================================================
st.markdown("""
<style>

/* ===== Global Dark Background ===== */
html, body, [data-testid="stApp"] {
    background: linear-gradient(
        135deg,
        #0b0f14,
        #0f172a,
        #020617
    ) !important;
    color: #e5e7eb !important;
}

/* ===== Main container ===== */
.block-container {
    padding-top: 2rem !important;
    background: linear-gradient(
        135deg,
        #0b0f14,
        #0f172a,
        #020617
    ) !important;
}

/* ===== Remove divider ===== */
hr {
    display: none !important;
}

/* ===== Cards ===== */
.card {
    background: linear-gradient(
        180deg,
        #111827,
        #0b1220
    ) !important;
    border-radius: 20px;
    padding: 30px;
    box-shadow: 0 12px 35px rgba(0, 0, 0, 0.6);
    border: 1px solid rgba(255, 255, 255, 0.05);
    margin-bottom: 30px;
}

/* ===== Titles ===== */
h1 {
    color: #ffffff !important;
    font-weight: 800;
    letter-spacing: 0.6px;
}

h2, h3, label {
    color: #cbd5f5 !important;
}

/* ===== Inputs ===== */
input {
    background: #111827 !important;
    color: #ffffff !important;
    border-radius: 12px !important;
    border: 1px solid #1f2937 !important;
}

/* ===== Buttons ===== */
.stButton > button {
    background: linear-gradient(90deg, #2563eb, #1d4ed8) !important;
    color: white !important;
    height: 54px;
    width: 100%;
    border-radius: 14px;
    font-size: 17px;
    font-weight: 700;
    border: none;
    transition: 0.25s ease;
}

.stButton > button:hover {
    transform: scale(1.03);
    background: linear-gradient(90deg, #1d4ed8, #2563eb) !important;
}

/* ===== Result Boxes ===== */
.success {
    background: rgba(16, 185, 129, 0.15);
    border-left: 6px solid #10b981;
    padding: 22px;
    border-radius: 14px;
    color: #d1fae5;
}

.danger {
    background: rgba(239, 68, 68, 0.18);
    border-left: 6px solid #ef4444;
    padding: 22px;
    border-radius: 14px;
    color: #fee2e2;
}

</style>
""", unsafe_allow_html=True)

# =====================================================
# Header
# =====================================================
st.markdown("## 💳 BODY Fraud Detection HUB")
st.markdown("### AI-Powered Credit Card Fraud Risk Assessment")

# =====================================================
# Input Section
# =====================================================
st.markdown('<div class="card">', unsafe_allow_html=True)
st.subheader("Transaction Details")

c1, c2, c3 = st.columns(3)

with c1:
    idx = st.number_input("Transaction ID", value=0)
    cc = st.number_input("Credit Card Number", value=0)
    amt = st.number_input("Transaction Amount ($)", value=100.0)

with c2:
    zipc = st.number_input("ZIP Code", value=0)
    lat = st.number_input("Customer Latitude", value=0.0)
    lon = st.number_input("Customer Longitude", value=0.0)

with c3:
    pop = st.number_input("City Population", value=100000)
    time = st.number_input("Unix Time", value=0)
    mlat = st.number_input("Merchant Latitude", value=0.0)
    mlon = st.number_input("Merchant Longitude", value=0.0)

st.markdown("</div>", unsafe_allow_html=True)

# =====================================================
# Prediction
# =====================================================
if st.button("🔍 Analyze Transaction Risk"):
    df = pd.DataFrame([[
        idx, cc, amt, zipc, lat, lon,
        pop, time, mlat, mlon
    ]], columns=FEATURES)

    x = scaler.transform(df)
    prob = model.predict_proba(x)[0][1]
    pred = 1 if prob >= THRESHOLD else 0

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("Risk Assessment Result")

    if pred == 1:
        st.markdown(f"""
        <div class="danger">
            <h3>🚨 High Fraud Risk Detected</h3>
            <p><b>Risk Probability:</b> {prob:.2%}</p>
            <p><b>Recommended Action:</b> Manual review required</p>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="success">
            <h3>✅ Transaction Appears Legitimate</h3>
            <p><b>Risk Probability:</b> {prob:.2%}</p>
            <p>No immediate action required</p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)

# Author: Gulam N Chabbi
# Project: Skin Cancer Detection using AI

import streamlit as st
from transformers import pipeline
from PIL import Image
import pandas as pd
import webbrowser

def set_bg():
    image_url = "https://plus.unsplash.com/premium_photo-1672759455907-bdaef741cd88?w=1920"

    st.markdown(f"""
    <style>

    /* FULL APP BACKGROUND */
    .stApp {{
        background-image: url("{image_url}");
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
        background-attachment: fixed;
    }}

    /* REMOVE WHITE DEFAULT BACKGROUND */
    [data-testid="stAppViewContainer"] {{
        background: transparent;
    }}

    /* MAIN CONTENT GLASS EFFECT */
    .block-container {{
        background: rgba(255, 255, 255, 0.85);
        padding: 25px;
        border-radius: 15px;
    }}

    /* SIDEBAR */
    [data-testid="stSidebar"] {{
        background: rgba(0,0,0,0.65);
    }}

    </style>
    """, unsafe_allow_html=True)
# --- PAGE CONFIG ---
st.set_page_config(
    page_title="Skin Disease Detection | Gulam N Chabbi",
    page_icon="⚕️",
    layout="wide"
)
set_bg()

# 👉 APPLY BACKGROUND
set_bg()


# --- MEDICAL DATABASE ---
MEDICAL_DB = {
    "Melanoma": {
        "severity": "critical",
        "risk_label": "🔴 MALIGNANT / CRITICAL",
        "description": "Dangerous skin cancer",
        "features": "• Irregular shape\n• Multiple colors",
        "causes": "UV radiation damage",
        "treatment": "Immediate surgery",
        "action": "🚨 See doctor immediately"
    },
    "Benign Keratosis": {
        "severity": "low",
        "risk_label": "SAFE",
        "description": "Non-cancerous growth",
        "features": "• Waxy texture",
        "causes": "Aging",
        "treatment": "No treatment needed",
        "action": "Safe"
    }
}

# --- MODEL ---
@st.cache_resource
def load_model():
    return pipeline("image-classification", model="Anwarkh1/Skin_Cancer-Image_Classification")


# --- SIDEBAR ---
with st.sidebar:
    st.title("⚙️ Controls")
    confidence_threshold = st.slider("Confidence (%)", 0, 100, 45)
    if st.button("Reset"):
        st.session_state.clear()
        st.rerun()


# --- MAIN UI ---
st.title("🏥 Skin Disease Detection AI")
st.caption("Developed by Gulam N Chabbi")

tab1, tab2, tab3 = st.tabs(["🔍 Scanner", "📚 Info", "📍 Help"])

# --- TAB 1: SCANNER ---
with tab1:
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Upload Image")
        img_file = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"])

        if img_file:
            img = Image.open(img_file)
            st.image(img, use_container_width=True)

            if st.button("Analyze"):
                with st.spinner("Analyzing..."):
                    model = load_model()
                    results = model(img)
                    st.session_state['results'] = results

    with col2:
        st.subheader("Results")

        if 'results' in st.session_state:
            top = st.session_state['results'][0]
            score = top['score'] * 100
            label = top['label'].replace('_', ' ').title()

            if score < confidence_threshold:
                st.error(f"Low confidence ({score:.2f}%)")
            else:
                info = MEDICAL_DB.get(label, MEDICAL_DB["Benign Keratosis"])

                # Result UI
                st.success(f"Detected: {label}")
                st.metric("Confidence", f"{score:.2f}%")

                st.markdown("### 📋 Details")
                st.write(info['description'])
                st.write(info['features'])

                st.markdown("### 💊 Treatment")
                st.info(info['treatment'])

                st.markdown("### ⚠️ Action")
                st.warning(info['action'])

                # Chart
                chart_data = pd.DataFrame([
                    {"Condition": r['label'], "Probability": r['score']*100}
                    for r in st.session_state['results'][:3]
                ])
                st.bar_chart(chart_data.set_index("Condition"))
        else:
            st.info("Upload an image to start analysis")


# --- TAB 2: INFO ---
with tab2:
    st.header("📚 Disease Info")

    cond = st.selectbox("Select Disease", list(MEDICAL_DB.keys()))
    data = MEDICAL_DB[cond]

    st.subheader(cond)
    st.write(data['description'])
    st.write(data['features'])

    col1, col2 = st.columns(2)
    with col1:
        st.write("### Causes")
        st.write(data['causes'])
    with col2:
        st.write("### Treatment")
        st.write(data['treatment'])

    st.warning(data['action'])


# --- TAB 3: HELP ---
with tab3:
    st.header("📍 Find Dermatologist")

    st.write("Click below to find nearby specialists")

    if st.button("Open Google Maps"):
        webbrowser.open("https://www.google.com/maps/search/dermatologist+near+me")

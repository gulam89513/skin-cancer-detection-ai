import streamlit as st
from transformers import pipeline
from PIL import Image
import pandas as pd
import webbrowser

# --- 1. PAGE CONFIGURATION ---
st.set_page_config(
    page_title="MediScan AI",
    page_icon="⚕️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- 2. MEDICAL DATABASE ---
MEDICAL_DB = {
    "Actinic Keratoses": {"severity": "high", "desc": "Pre-cancerous sun damage.", "action": "Consult dermatologist."},
    "Basal Cell Carcinoma": {"severity": "high", "desc": "Common skin cancer.", "action": "Schedule a biopsy."},
    "Benign Keratosis": {"severity": "low", "desc": "Age-related growth.", "action": "Monitor for changes."},
    "Dermatofibroma": {"severity": "low", "desc": "Non-cancerous firm bump.", "action": "Usually harmless."},
    "Melanocytic Nevi": {"severity": "low", "desc": "Common mole.", "action": "Monitor using ABCDE rule."},
    "Melanoma": {"severity": "critical", "desc": "Serious skin cancer.", "action": "🚨 SEE A DOCTOR IMMEDIATELY."},
    "Vascular Lesions": {"severity": "low", "desc": "Blood vessel growth.", "action": "Usually harmless."}
}

# --- 3. MODEL LOADING ---
@st.cache_resource
def load_model():
    return pipeline("image-classification", model="Anwarkh1/Skin_Cancer-Image_Classification")

# --- 4. SIDEBAR (ONLY ANALYSIS CONTROLS) ---
with st.sidebar:
    st.title("⚙️ Controls")
    st.divider()
    # Move your controls here
    confidence_threshold = st.slider("Sensitivity Threshold (%)", 0, 100, 30, help="Filters out low-confidence results.")
    st.divider()
    st.caption("Developed by Gulam")
    st.info("ℹ️ Your images are processed in memory and not stored.")

# --- 5. MAIN PAGE NAVIGATION (USING TABS) ---
st.title("🏥 MediScan AI")
tab_scan, tab_dict, tab_help = st.tabs(["🔍 Patient Scan", "📚 Medical Dictionary", "🚑 Emergency Help"])

# --- TAB 1: SCANNER ---
with tab_scan:
    st.subheader("🩺 AI Skin Detection")
    st.warning("DISCLAIMER: For educational purposes only.")
    
    col1, col2 = st.columns([1, 1.2])
    
    with col1:
        img_file = st.file_uploader("Upload lesion image", type=["jpg", "png", "jpeg"])
        if img_file:
            img = Image.open(img_file)
            st.image(img, use_container_width=True)
            
            if st.button("🚀 Analyze Now", type="primary"):
                with st.spinner("Analyzing neural network..."):
                    model = load_model()
                    results = model(img)
                    st.session_state['results'] = results

    with col2:
        if 'results' in st.session_state:
            res = st.session_state['results'][0]
            score = res['score'] * 100
            
            # Check against sidebar threshold
            if score < confidence_threshold:
                st.error(f"Result below {confidence_threshold}% threshold. Please provide a clearer image.")
            else:
                label = res['label'].replace('_', ' ').title()
                info = MEDICAL_DB.get(label, {"severity": "low", "action": "Consult doctor.", "desc": "N/A"})
                
                if info['severity'] == "critical": st.error(f"DETECTION: {label.upper()}")
                elif info['severity'] == "high": st.warning(f"DETECTION: {label.upper()}")
                else: st.success(f"DETECTION: {label.upper()}")
                
                st.metric("Confidence Score", f"{score:.2f}%")
                st.write(f"**Description:** {info['desc']}")
                st.write(f"**Action Plan:** {info['action']}")
                
                # Probability Chart
                chart_data = pd.DataFrame([{"Condition": r['label'].title(), "Prob": r['score']*100} for r in st.session_state['results']])
                st.bar_chart(chart_data.set_index("Condition"))
        else:
            st.info("Upload an image to see the diagnostic report.")

# --- TAB 2: DICTIONARY ---
with tab_dict:
    st.header("📚 Condition Database")
    selected = st.selectbox("Select condition:", list(MEDICAL_DB.keys()))
    data = MEDICAL_DB[selected]
    st.subheader(selected)
    st.write(data['desc'])
    st.info(f"Recommended Step: {data['action']}")

# --- TAB 3: EMERGENCY ---
with tab_help:
    st.header("🚑 Find Assistance")
    st.write("If you received a high-risk result, locate a specialist near you.")
    if st.button("🔍 Open Google Maps"):
        webbrowser.open_new_tab("https://www.google.com/maps/search/dermatologist+near+me")

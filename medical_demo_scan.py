import streamlit as st
from transformers import pipeline
from PIL import Image
import pandas as pd
import webbrowser

# --- 1. FORCE SIDEBAR OPEN ---
st.set_page_config(
    page_title="MediScan AI",
    page_icon="⚕️",
    layout="wide",
    initial_sidebar_state="expanded" # This tells Streamlit to show the sidebar on load
)

# --- 2. PROFESSIONAL MEDICAL DATA ---
MEDICAL_DB = {
    "Melanoma": {"severity": "critical", "action": "🚨 SEE A DOCTOR IMMEDIATELY.", "desc": "Most serious skin cancer."},
    "Basal Cell Carcinoma": {"severity": "high", "action": "Schedule a biopsy.", "desc": "Common skin cancer, locally invasive."},
    "Actinic Keratoses": {"severity": "high", "action": "Consult dermatologist.", "desc": "Pre-cancerous sun damage."},
    "Melanocytic Nevi": {"severity": "low", "action": "Monitor changes.", "desc": "Common mole."},
    "Benign Keratosis": {"severity": "low", "action": "Usually safe.", "desc": "Age-related growth."},
    "Dermatofibroma": {"severity": "low", "action": "No action needed.", "desc": "Non-cancerous bump."},
    "Vascular Lesions": {"severity": "low", "action": "Usually harmless.", "desc": "Blood vessel growth."}
}

# --- 3. LOAD AI MODEL ---
@st.cache_resource
def load_model():
    return pipeline("image-classification", model="Anwarkh1/Skin_Cancer-Image_Classification")

# --- 4. THE SIDEBAR (ALWAYS VISIBLE) ---
with st.sidebar:
    st.title("🏥 MediScan AI")
    st.divider()
    # The Navigation Menu
    page = st.radio("Navigation", ["🔍 Scanner", "📚 Dictionary", "🚑 Help"])
    st.divider()
    st.caption("Developed by Gulam")
    st.info("Tip: If the menu disappears on mobile, click the '>' in the top left.")

# --- 5. PAGE ROUTING ---

# SCANNER PAGE
if page == "🔍 Scanner":
    st.title("🩺 AI Skin Detection")
    st.markdown("---")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("1. Upload")
        img_file = st.file_uploader("Upload lesion image", type=["jpg", "png", "jpeg"])
        if img_file:
            img = Image.open(img_file)
            st.image(img, use_container_width=True)
            
            if st.button("🚀 Analyze Now", type="primary"):
                with st.spinner("Analyzing..."):
                    model = load_model()
                    results = model(img)
                    st.session_state['results'] = results

    with col2:
        st.subheader("2. Results")
        if 'results' in st.session_state:
            top = st.session_state['results'][0]
            name = top['label'].replace('_', ' ').title()
            score = top['score'] * 100
            info = MEDICAL_DB.get(name, {"severity": "low", "action": "Consult doctor.", "desc": "N/A"})
            
            # Show Alert based on severity
            if info['severity'] == "critical": st.error(f"DETECTION: {name}")
            elif info['severity'] == "high": st.warning(f"DETECTION: {name}")
            else: st.success(f"DETECTION: {name}")
            
            st.metric("AI Confidence", f"{score:.1f}%")
            st.write(f"**What it is:** {info['desc']}")
            st.write(f"**Next Step:** {info['action']}")
        else:
            st.info("Upload an image and click 'Analyze' to see results.")

# DICTIONARY PAGE
elif page == "📚 Dictionary":
    st.title("📚 Medical Dictionary")
    choice = st.selectbox("Select a condition to learn more:", list(MEDICAL_DB.keys()))
    data = MEDICAL_DB[choice]
    st.subheader(choice)
    st.write(data['desc'])
    st.info(f"Recommended Action: {data['action']}")

# HELP PAGE
elif page == "🚑 Help":
    st.title("🚑 Find a Specialist")
    st.write("Locate a dermatologist near your current location.")
    if st.button("🔍 Search on Google Maps"):
        webbrowser.open_new_tab("https://www.google.com/maps/search/dermatologist+near+me")

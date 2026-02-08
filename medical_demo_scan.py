import streamlit as st
from transformers import pipeline
from PIL import Image
import pandas as pd
import webbrowser
import time

# --- 1. PROFESSIONAL PAGE CONFIGURATION ---
# Note: st.set_page_config must be the very first Streamlit command
st.set_page_config(
    page_title="MediScan AI | Clinical Decision Support",
    page_icon="⚕️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- 2. HIDE STREAMLIT BRANDING ---
hide_st_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            header {visibility: hidden;}
            .stApp > header {display: none;}
            </style>
            """
st.markdown(hide_st_style, unsafe_allow_html=True)

# --- 3. MEDICAL KNOWLEDGE BASE ---
MEDICAL_DB = {
    "Actinic Keratoses": {
        "severity": "high",
        "description": "Also known as solar keratosis, a rough, scaly patch on the skin caused by years of sun exposure.",
        "causes": "Long-term exposure to UV radiation. Pre-cancerous condition.",
        "treatment": "Cryotherapy, topical creams, or laser therapy.",
        "action": "⚠️ Consult a dermatologist. Can turn into squamous cell carcinoma."
    },
    "Basal Cell Carcinoma": {
        "severity": "high",
        "description": "A type of skin cancer that begins in the basal cells. Often appears as a transparent bump.",
        "causes": "Intense sun exposure and UV radiation.",
        "treatment": "Surgical excision, Mohs surgery, or electrosurgery.",
        "action": "🚨 Schedule a biopsy. Significant local damage if ignored."
    },
    "Benign Keratosis": {
        "severity": "low",
        "description": "Non-cancerous skin growths that often appear with age.",
        "causes": "Genetics and age. Not caused by sun exposure.",
        "treatment": "No treatment medically necessary.",
        "action": "✅ Generally safe. Monitor for changes."
    },
    "Dermatofibroma": {
        "severity": "low",
        "description": "A common, non-cancerous skin growth appearing as a firm bump.",
        "causes": "Minor skin injury, bug bites, or splinters.",
        "treatment": "Usually requires no treatment.",
        "action": "✅ Benign. No action needed unless it bleeds."
    },
    "Melanocytic Nevi": {
        "severity": "low",
        "description": "Commonly known as a 'Mole'. A benign proliferation of pigment cells.",
        "causes": "Clusters of pigment cells.",
        "treatment": "No treatment needed unless cancer is suspected.",
        "action": "✅ Monitor using the ABCDE rule."
    },
    "Melanoma": {
        "severity": "critical",
        "description": "The most serious type of skin cancer. Develops in melanocytes.",
        "causes": "DNA damage from UV radiation and genetics.",
        "treatment": "Immediate surgical removal; possible immunotherapy.",
        "action": "🚨 URGENT: See a dermatologist immediately for a biopsy."
    },
    "Vascular Lesions": {
        "severity": "low",
        "description": "Abnormalities of blood vessels like cherry angiomas.",
        "causes": "Aging, sun exposure, or hormonal changes.",
        "treatment": "Laser therapy for cosmetic removal.",
        "action": "✅ Usually harmless."
    }
}

# --- 4. MODEL LOADING ---
@st.cache_resource
def load_model():
    # Loading the specified HuggingFace model
    return pipeline("image-classification", model="Anwarkh1/Skin_Cancer-Image_Classification")

# --- 5. SIDEBAR ---
with st.sidebar:
    st.title("🏥 MediScan AI")
    st.caption("Professional Dermatological Assistant")
    st.markdown("---")
    st.subheader("⚙️ Analysis Controls")
    confidence_threshold = st.slider("Sensitivity Threshold (%)", 0, 100, 30)
    st.info("ℹ️ **Privacy:** Images processed locally and not stored.")

# --- 6. MAIN INTERFACE ---
st.title("🩺 AI Skin Detection")
st.markdown("""
<div style='background-color: rgba(33, 150, 243, 0.1); padding: 15px; border-radius: 10px; border-left: 5px solid #2196f3;'>
    <strong>CLINICAL DISCLAIMER:</strong> For educational purposes only. Does NOT replace a professional diagnosis.
</div>
""", unsafe_allow_html=True)

tab_scan, tab_info, tab_help = st.tabs(["🔍 Patient Scan", "📚 Medical Dictionary", "🚑 Emergency/Help"])

# --- TAB 1: SCANNER ---
with tab_scan:
    col_input, col_results = st.columns([1, 1.5])

    with col_input:
        st.subheader("1. Specimen Acquisition")
        source = st.radio("Input Source:", ["Upload File", "Live Camera"], horizontal=True)
        
        img_file = st.file_uploader("Upload image", type=['png', 'jpg', 'jpeg']) if source == "Upload File" else st.camera_input("Capture")

        if img_file:
            img = Image.open(img_file)
            st.image(img, caption="Target Lesion", use_container_width=True)
            if st.button("🚀 Run Analysis", type="primary"):
                with st.spinner('Analyzing neural network...'):
                    classifier = load_model()
                    st.session_state['results'] = classifier(img)
                    st.session_state['image_processed'] = True

    with col_results:
        st.subheader("2. Diagnostic Report")
        if 'image_processed' in st.session_state:
            results = st.session_state['results']
            res = results[0]
            label = res['label'].replace('_', ' ').title()
            score = res['score'] * 100
            
            info = MEDICAL_DB.get(label, MEDICAL_DB["Vascular Lesions"])
            
            if info['severity'] == "critical": st.error(f"⚠️ {label.upper()}")
            elif info['severity'] == "high": st.warning(f"⚠️ {label.upper()}")
            else: st.success(f"✅ {label.upper()}")

            st.metric("Confidence Score", f"{score:.2f}%")
            
            with st.expander("📖 Description", expanded=True): st.write(info['description'])
            with st.expander("🛡️ Action Plan"): st.write(info['action'])
            
            # Probability Chart
            chart_data = pd.DataFrame([
                {"Condition": r['label'].replace('_', ' ').title(), "Prob": r['score']*100} 
                for r in results
            ])
            st.bar_chart(chart_data.set_index("Condition"))
        else:
            st.info("Upload an image to see the report.")

# --- TAB 2: DICTIONARY ---
with tab_info:
    st.header("Medical Knowledge Base")
    selected = st.selectbox("Select Condition:", list(MEDICAL_DB.keys()))
    data = MEDICAL_DB[selected]
    st.subheader(selected)
    st.write(f"**Definition:** {data['description']}")
    st.write(f"**Causes:** {data['causes']}")
    st.info(f"**Treatment:** {data['treatment']}")

# --- TAB 3: EMERGENCY ---
with tab_help:
    st.header("Find a Specialist")
    if st.button("🔍 Open Google Maps for Dermatologists"):
        webbrowser.open_new_tab("https://www.google.com/maps/search/dermatologist")

import streamlit as st
from transformers import pipeline
from PIL import Image
import pandas as pd
import webbrowser
import time

# --- 1. PROFESSIONAL PAGE CONFIGURATION ---
st.set_page_config(
    page_title="MediScan AI | Clinical Decision Support",
    page_icon="⚕️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- 2. MEDICAL KNOWLEDGE BASE (THE BRAIN) ---
# This dictionary contains the professional medical info for each class
MEDICAL_DB = {
    "Actinic Keratoses": {
        "severity": "high",
        "description": "Also known as solar keratosis, this is a rough, scaly patch on the skin caused by years of sun exposure.",
        "causes": "Long-term exposure to ultraviolet (UV) radiation from sunlight or tanning beds. It is considered a pre-cancerous condition.",
        "treatment": "Treatments typically include Cryotherapy (freezing), topical creams (5-fluorouracil), or laser therapy to remove the damaged cells.",
        "action": "⚠️ Consult a dermatologist. Untreated AKs can turn into squamous cell carcinoma."
    },
    "Basal Cell Carcinoma": {
        "severity": "high",
        "description": "A type of skin cancer that begins in the basal cells. It often appears as a slightly transparent bump on the skin.",
        "causes": "Intense sun exposure and UV radiation. It is the most common form of skin cancer but rarely spreads to other parts of the body.",
        "treatment": "Common treatments include surgical excision, Mohs surgery, or electrosurgery. Prognosis is generally excellent if treated early.",
        "action": "🚨 Schedule a biopsy. While rarely fatal, it can cause significant local damage if ignored."
    },
    "Benign Keratosis": {
        "severity": "low",
        "description": "Often called Seborrheic Keratosis. These are non-cancerous skin growths that often appear with age.",
        "causes": "The exact cause is unknown, but genetics and age play a role. They are NOT caused by sun exposure and are not contagious.",
        "treatment": "No treatment is medically necessary. If they are irritated or for cosmetic reasons, they can be removed via cryotherapy.",
        "action": "✅ Generally safe. Monitor for changes in shape or color, but usually no urgent action is needed."
    },
    "Dermatofibroma": {
        "severity": "low",
        "description": "A common, non-cancerous (benign) skin growth. It typically appears as a firm, small bump.",
        "causes": "Often develops after a minor skin injury, such as a bug bite, splinter, or prick.",
        "treatment": "Harmless and usually requires no treatment. Surgical removal is possible if it becomes painful.",
        "action": "✅ Benign. No action needed unless the spot changes or bleeds."
    },
    "Melanocytic Nevi": {
        "severity": "low",
        "description": "Commonly known as a 'Mole'. It is a benign proliferation of melanocytes (pigment cells).",
        "causes": "Caused by clusters of pigment cells. Most adults have between 10 and 40 common moles.",
        "treatment": "No treatment needed for common moles. Removal is done only for cosmetic reasons or if cancer is suspected.",
        "action": "✅ Monitor using the ABCDE rule. If it changes size/color, see a doctor."
    },
    "Melanoma": {
        "severity": "critical",
        "description": "The most serious type of skin cancer. It develops in the cells (melanocytes) that produce melanin.",
        "causes": "DNA damage from UV radiation (sun/tanning) triggers mutations. Genetics also play a significant role.",
        "treatment": "Requires immediate surgical removal (wide local excision). Advanced stages may need immunotherapy, radiation, or chemotherapy.",
        "action": "🚨 URGENT: See a dermatologist immediately for a biopsy. Early detection is life-saving."
    },
    "Vascular Lesions": {
        "severity": "low",
        "description": "Abnormalities of blood vessels or other vessels. Includes cherry angiomas or spider veins.",
        "causes": "Can be congenital (birthmarks) or acquired due to aging, sun exposure, or hormonal changes.",
        "treatment": "Laser therapy is the most common treatment for cosmetic removal.",
        "action": "✅ Usually harmless. Consult a doctor if the lesion bleeds or grows rapidly."
    }
}

# --- 3. CUSTOM STYLING (CSS) ---
st.markdown("""
    <style>
    .big-font { font-size:20px !important; }
    .stAlert { padding: 15px; border-radius: 10px; }
    .metric-card { background-color: #ffffff; padding: 20px; border-radius: 10px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); text-align: center;}
    </style>
    """, unsafe_allow_html=True)

# --- 4. MODEL LOADING ---
@st.cache_resource
def load_model():
    return pipeline("image-classification", model="Anwarkh1/Skin_Cancer-Image_Classification")

# --- 5. SIDEBAR ---
with st.sidebar:
    st.title("🏥 MediScan AI")
    st.caption("Professional Dermatological Assistant")
    
    st.markdown("---")
    st.subheader("⚙️ Analysis Controls")
    confidence_threshold = st.slider("Sensitivity Threshold (%)", 0, 100, 30, help="Minimum confidence required to display a result.")
    
    st.markdown("---")
    st.info("ℹ️ **Privacy Note:** \nYour images are processed locally in memory and are not saved to any server.")

# --- 6. MAIN INTERFACE ---
st.title("🩺 AI Skin Lesion Analysis System")
st.markdown("""
<div style='background-color: #e3f2fd; padding: 15px; border-radius: 10px; border-left: 5px solid #2196f3;'>
    <strong>CLINICAL DISCLAIMER:</strong> This tool utilizes Artificial Intelligence to screen for potential skin pathologies. 
    It is intended for <strong>educational and screening purposes only</strong>. It does <strong>NOT</strong> replace a professional biopsy or diagnosis by a certified dermatologist.
</div>
""", unsafe_allow_html=True)

# Tabs
tab_scan, tab_info, tab_help = st.tabs(["🔍 Patient Scan", "📚 Medical Dictionary", "🚑 Emergency/Help"])

# --- TAB 1: SCANNER ---
with tab_scan:
    col_input, col_results = st.columns([1, 1.5])

    # LEFT COLUMN: INPUT
    with col_input:
        st.subheader("1. Specimen Acquisition")
        source = st.radio("Select Input Source:", ["Upload Image File", "Live Camera Capture"], horizontal=True)
        
        img_file = None
        if source == "Upload Image File":
            img_file = st.file_uploader("Upload dermatoscopic image", type=['png', 'jpg', 'jpeg'])
        else:
            img_file = st.camera_input("Capture lesion area")

        if img_file:
            img = Image.open(img_file)
            st.image(img, caption="Loaded Specimen", use_container_width=True)
            
            # RUN BUTTON
            if st.button("🚀 Initiate Analysis", type="primary"):
                with st.spinner('Processing neural network layers...'):
                    # Simulate processing time for "professional feel"
                    time.sleep(1) 
                    
                    classifier = load_model()
                    results = classifier(img)
                    
                    # Store results in session state to persist them
                    st.session_state['results'] = results
                    st.session_state['image_processed'] = True

    # RIGHT COLUMN: RESULTS
    with col_results:
        st.subheader("2. Diagnostic Report")
        
        if 'image_processed' in st.session_state and st.session_state['image_processed']:
            results = st.session_state['results']
            
            # Extract top result
            top_result = results[0]
            label_raw = top_result['label']
            score = top_result['score'] * 100
            
            # Format label
            label_clean = label_raw.replace('_', ' ').title()
            
            # Get Medical Data
            med_info = MEDICAL_DB.get(label_clean, {
                "severity": "unknown",
                "description": "No specific data available for this class.",
                "causes": "Unknown.",
                "treatment": "Consult a doctor.",
                "action": "Consult a doctor."
            })
            
            # --- DYNAMIC UI BASED ON SEVERITY ---
            severity_color = "blue"
            if med_info['severity'] == "critical":
                severity_color = "red"
                st.error(f"⚠️ DETECTION: {label_clean.upper()}")
            elif med_info['severity'] == "high":
                severity_color = "orange"
                st.warning(f"⚠️ DETECTION: {label_clean.upper()}")
            else:
                severity_color = "green"
                st.success(f"✅ DETECTION: {label_clean.upper()}")

            # Confidence Metric
            st.metric(label="AI Confidence Score", value=f"{score:.2f}%")
            
            st.divider()
            
            # --- DETAILED MEDICAL INSIGHTS ---
            st.markdown("### 📋 Clinical Insights")
            
            with st.expander("📖 What is this condition?", expanded=True):
                st.write(med_info['description'])
                
            with st.expander("🧬 Etiology (Causes & Risk Factors)"):
                st.write(med_info['causes'])
                
            with st.expander("💊 Standard Treatments & Management"):
                st.info(med_info['treatment'])
                
            with st.expander("🛡️ Recommended Action Plan"):
                st.markdown(f"**Status:** {med_info['severity'].upper()} RISK")
                st.write(med_info['action'])

            # Full Probability Chart
            st.divider()
            st.caption("Differential Diagnosis Probabilities")
            chart_data = pd.DataFrame([{
                "Condition": res['label'].replace('_', ' ').title(),
                "Probability": res['score'] * 100
            } for res in results])
            st.bar_chart(chart_data.set_index("Condition"))

        else:
            st.info("Waiting for input data... upload an image to begin.")

# --- TAB 2: DICTIONARY ---
with tab_info:
    st.header("Medical Knowledge Base")
    st.write("Browse the database of skin conditions trained into this model.")
    
    selected_condition = st.selectbox("Select a condition to learn more:", list(MEDICAL_DB.keys()))
    
    info = MEDICAL_DB[selected_condition]
    st.subheader(selected_condition)
    st.write(f"**Definition:** {info['description']}")
    st.write(f"**Causes:** {info['causes']}")
    st.write(f"**Treatments:** {info['treatment']}")

# --- TAB 3: FIND HELP ---
with tab_help:
    st.header("Find a Specialist")
    st.write("If your results indicate High or Critical risk, please locate a dermatologist immediately.")
    
    if st.button("🔍 Find Dermatologists on Google Maps"):
        webbrowser.open_new_tab("https://www.google.com/maps/search/dermatologist+near+me")
        st.success("Redirecting to Google Maps...")
    
    st.markdown("---")
    st.warning("⚠️ **IMPORTANT:** Do not attempt to self-treat based on this AI analysis. Always seek professional medical validation.")
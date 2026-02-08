import streamlit as st
from transformers import pipeline
from PIL import Image
import pandas as pd
import webbrowser

# --- 1. PAGE CONFIGURATION ---
st.set_page_config(
    page_title="MediScan AI | Clinical Decision Support",
    page_icon="⚕️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- 2. FULL MEDICAL KNOWLEDGE BASE (RESTORED) ---
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

# --- 3. MODEL LOADING ---
@st.cache_resource
def load_model():
    return pipeline("image-classification", model="Anwarkh1/Skin_Cancer-Image_Classification")

# --- 4. SIDEBAR (CONTROLS) ---
with st.sidebar:
    st.title("⚙️ Analysis Controls")
    st.divider()
    
    # FILTER: Default set to 50% to filter out "Books/Random Objects"
    confidence_threshold = st.slider(
        "Confidence Threshold (%)", 
        0, 100, 50, 
        help="If the AI is not sure (low score), it will refuse to diagnose. This helps filter out non-skin images."
    )
    
    st.divider()
    st.caption("Developed by Gulam")
    
    # RESET BUTTON
    if st.button("🔄 Reset App"):
        st.session_state.clear()
        st.rerun()

# --- 5. MAIN INTERFACE ---
st.title("🏥 MediScan AI")
tab_scan, tab_dict, tab_help = st.tabs(["🔍 Patient Scan", "📚 Medical Dictionary", "🚑 Emergency Help"])

# --- TAB 1: SCANNER ---
with tab_scan:
    st.subheader("🩺 AI Skin Detection")
    st.info("⚠️ NOTE: This AI is designed for **SKIN ONLY**. Uploading objects (books, faces, cars) will result in errors or inconclusive results.")
    
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
            # Get the top result
            top_result = st.session_state['results'][0]
            score_percent = top_result['score'] * 100
            label = top_result['label'].replace('_', ' ').title()
            
            # THE FIX: FILTER OUT LOW CONFIDENCE IMAGES (e.g., Books)
            if score_percent < confidence_threshold:
                st.error("⚠️ INCONCLUSIVE RESULT")
                st.warning(f"Confidence: {score_percent:.1f}% (Below Threshold of {confidence_threshold}%)")
                st.write("The AI is not confident this matches a known skin condition.")
                st.info("💡 **Tip:** This often happens if the image is blurry, dark, or **not a skin lesion** (e.g., an object).")
            else:
                # If high confidence, show the full diagnosis with RESTORED DATA
                info = MEDICAL_DB.get(label, {
                    "severity": "unknown", 
                    "description": "N/A", 
                    "causes": "N/A", 
                    "treatment": "N/A", 
                    "action": "Consult doctor."
                })
                
                if info['severity'] == "critical": st.error(f"DETECTION: {label.upper()}")
                elif info['severity'] == "high": st.warning(f"DETECTION: {label.upper()}")
                else: st.success(f"DETECTION: {label.upper()}")
                
                st.metric("AI Confidence Score", f"{score_percent:.2f}%")
                
                # --- DETAILED MEDICAL INSIGHTS (RESTORED) ---
                st.markdown("### 📋 Clinical Insights")
                
                with st.expander("📖 What is this condition?", expanded=True):
                    st.write(info['description'])
                    
                with st.expander("🧬 Causes & Risk Factors"):
                    st.write(info['causes'])
                    
                with st.expander("💊 Standard Treatments & Management"):
                    st.info(info['treatment'])
                    
                with st.expander("🛡️ Recommended Action Plan"):
                    st.markdown(f"**Status:** {info['severity'].upper()} RISK")
                    st.write(info['action'])
                
                # Show Chart
                st.divider()
                st.caption("Differential Diagnosis Probabilities")
                chart_data = pd.DataFrame([{"Condition": r['label'].replace('_', ' ').title(), "Prob": r['score']*100} for r in st.session_state['results']])
                st.bar_chart(chart_data.set_index("Condition"))
        else:
            st.write("Waiting for upload...")

# --- TAB 2: DICTIONARY ---
with tab_dict:
    st.header("📚 Condition Database")
    selected = st.selectbox("Select condition:", list(MEDICAL_DB.keys()))
    data = MEDICAL_DB[selected]
    
    st.subheader(selected)
    st.write(f"**Definition:** {data['description']}")
    st.write(f"**Causes:** {data['causes']}")
    st.info(f"**Treatment:** {data['treatment']}")
    st.warning(f"**Action:** {data['action']}")

# --- TAB 3: EMERGENCY ---
with tab_help:
    st.header("🚑 Find Assistance")
    st.write("If you received a high-risk result, locate a specialist near you.")
    if st.button("🔍 Open Google Maps"):
        webbrowser.open_new_tab("https://www.google.com/maps/search/dermatologists+near+me")

# Author: Gulam N Chabbi
# Project: Skin Cancer Detection using AI
# Created: (put the month/year you actually started, e.g. Jan 2026)
# GitHub: gulam89513
# NOTE: This repository was made public earlier for learning/demo purposes.
# Unauthorized academic submission or reuse without permission is prohibited.

import streamlit as st
from transformers import pipeline
from PIL import Image
import pandas as pd
import webbrowser

# --- 1. PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Skin Disease Detection | Developed by Gulam N Chabbi",
    page_icon="⚕️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- 2. ENHANCED MEDICAL DATABASE (Distinct & Specific) ---
MEDICAL_DB = {
    "Actinic Keratoses": {
        "severity": "high",
        "risk_label": "PRE-CANCEROUS / HIGH RISK",
        "description": "A rough, scaly patch on the skin caused by years of sun exposure.",
        "features": "• Sandpaper-like texture\n• Red, pink, or brown scaly patch\n• Itching or burning sensation",
        "causes": "☀️ **Specific Cause:** Cumulative UV damage from sunlight or tanning beds. The skin cells have been damaged over many years.",
        "treatment": "💊 **Treatment Protocol:** Cryotherapy (freezing), 5-fluorouracil cream, or chemical peels.",
        "action": "⚠️ **Consult Dermatologist:** These can turn into Squamous Cell Carcinoma if ignored."
    },
    "Basal Cell Carcinoma": {
        "severity": "high",
        "risk_label": "MALIGNANT / HIGH RISK",
        "description": "The most common form of skin cancer. It grows slowly and rarely spreads.",
        "features": "• Pearly or waxy bump\n• Visible blood vessels on the growth\n• A sore that bleeds, heals, and returns",
        "causes": "☀️ **Specific Cause:** Intense, intermittent sun exposure (like sunburns) causing DNA mutations in basal cells.",
        "treatment": "💊 **Treatment Protocol:** Mohs Surgery (gold standard), Excision, or Electrodessication.",
        "action": "🚨 **Schedule Biopsy:** Highly treatable if caught now. Do not wait."
    },
    "Benign Keratosis": {
        "severity": "low",
        "risk_label": "BENIGN / HARMLESS",
        "description": "A non-cancerous skin growth (Seborrheic Keratosis) common in older adults.",
        "features": "• Waxy, 'stuck-on' appearance\n• Well-defined borders\n• Tan, brown, or black color",
        "causes": "🧬 **Specific Cause:** Genetic aging process. These are NOT caused by sun and are NOT contagious.",
        "treatment": "✅ **Treatment Protocol:** None needed. Can be frozen off if it gets irritated by clothing.",
        "action": "✅ **Safe:** No action needed unless it changes shape rapidly."
    },
    "Dermatofibroma": {
        "severity": "low",
        "risk_label": "BENIGN / HARMLESS",
        "description": "A firm, non-cancerous bump that often forms after a minor injury.",
        "features": "• Firm, hard nodule under the skin\n• Dimples inward when pinched\n• Pink or brown color",
        "causes": "🐜 **Specific Cause:** Often scar tissue reacting to a bug bite, splinter, or shaving nick.",
        "treatment": "✅ **Treatment Protocol:** Harmless. Surgical removal leaves a scar, so doctors usually leave it alone.",
        "action": "✅ **Safe:** It may persist for years but is not dangerous."
    },
    "Melanocytic Nevi": {
        "severity": "low",
        "risk_label": "BENIGN / MONITOR REQUIRED",
        "description": "A common mole. A benign cluster of pigment cells.",
        "features": "• Uniform brown or black color\n• Round/Oval shape\n• Sharp, clean borders",
        "causes": "🧬 **Specific Cause:** Genetic clustering of melanocytes. Sun exposure in childhood increases the count.",
        "treatment": "✅ **Treatment Protocol:** No treatment. Removal is only for cosmetic reasons.",
        "action": "🔍 **Monitor:** Watch for the 'ABCDEs' (Asymmetry, Border, Color, Diameter, Evolving)."
    },
    "Melanoma": {
        "severity": "critical",
        "risk_label": "🔴 MALIGNANT / CRITICAL LIFE THREAT",
        "description": "The most dangerous skin cancer. Uncontrolled growth of pigment cells.",
        "features": "• ASYMMETRICAL shape\n• IRREGULAR, jagged borders\n• MULTIPLE colors (black, blue, red)\n• LARGER than a pencil eraser",
        "causes": "⚠️ **Specific Cause:** Severe DNA damage from UV rays triggering rapid, uncontrolled cell growth.",
        "treatment": "🚨 **Treatment Protocol:** IMMEDIATE wide excision surgery. May require immunotherapy or radiation.",
        "action": "🚨 **EMERGENCY:** See a doctor IMMEDIATELY. Early detection is vital for survival."
    },
    "Vascular Lesions": {
        "severity": "low",
        "risk_label": "BENIGN / HARMLESS",
        "description": "Abnormal bunching of blood vessels near the skin surface.",
        "features": "• Bright red or purple color\n• Turns white (blanches) when pressed\n• Soft to the touch",
        "causes": "🩸 **Specific Cause:** Aging (Cherry Angiomas), pregnancy hormones, or liver issues.",
        "treatment": "✅ **Treatment Protocol:** Laser therapy (Vascular Laser) if removal is desired.",
        "action": "✅ **Safe:** Usually harmless. See a doctor only if it bleeds extensively."
    }
}

# --- 3. MODEL LOADING ---
@st.cache_resource
def load_model():
    return pipeline("image-classification", model="Anwarkh1/Skin_Cancer-Image_Classification")

# --- 4. SIDEBAR ---
with st.sidebar:
    st.title("⚙️ MediScan Controls")
    st.divider()
    confidence_threshold = st.slider("Accuracy Threshold (%)", 0, 100, 45, help="Filters out unclear images.")
    st.divider()
    st.caption("Developed by Gulam N Chabbi")
    if st.button("🔄 Reset Analysis"):
        st.session_state.clear()
        st.rerun()

# --- 5. MAIN INTERFACE ---
st.title("🏥 Skin Disease Detection")
st.caption("Developed by Gulam N Chabbi")

tab_scan, tab_dict, tab_help = st.tabs(["🔍 Clinical Scanner", "📚 Disease Encyclopedia", "🚑 Specialist Locator"])

# --- TAB 1: SCANNER ---
with tab_scan:
    col1, col2 = st.columns([0.8, 1.2])
    
    with col1:
        st.subheader("1. Specimen Input")
        st.info("📸 **Guidance:** Ensure the disease image is centered and well-lit.")
        img_file = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"])
        
        if img_file:
            img = Image.open(img_file)
            st.image(img, caption="Analyzed Specimen", use_container_width=True)
            
            if st.button("🚀 Run Diagnostics", type="primary"):
                with st.spinner("Processing Neural Network Layers..."):
                    model = load_model()
                    results = model(img)
                    st.session_state['results'] = results

    with col2:
        st.subheader("2. Diagnostic Results")
        
        if 'results' in st.session_state:
            top = st.session_state['results'][0]
            score = top['score'] * 100
            label_raw = top['label']
            label = label_raw.replace('_', ' ').title()
            
            # --- FILTER LOGIC ---
            if score < confidence_threshold:
                st.error("⚠️ ANALYSIS INCONCLUSIVE")
                st.warning(f"Confidence Level: {score:.1f}% (Below required {confidence_threshold}%)")
                st.write("The AI is not confident. Please use a clearer image of a skin lesion.")
            else:
                info = MEDICAL_DB.get(label, {
                    "severity": "low", "risk_label": "UNKNOWN", "description": "N/A", 
                    "features": "N/A", "causes": "N/A", "treatment": "N/A", "action": "Consult doctor"
                })
                
                # --- RESULT HEADER ---
                if info['severity'] == "critical":
                    st.error(f"🔴 DETECTION: {label.upper()}")
                elif info['severity'] == "high":
                    st.warning(f"🟠 DETECTION: {label.upper()}")
                else:
                    st.success(f"🟢 DETECTION: {label.upper()}")

                st.write(f"**Risk Assessment:** {info['risk_label']}")
                st.metric("AI Confidence Probability", f"{score:.2f}%")
                
                st.divider()
                
                # --- CLINICAL BREAKDOWN (Distinct Sections) ---
                st.markdown("### 📋 Clinical Breakdown")
                
                # 1. VISUAL FEATURES (Specific to the disease)
                with st.expander("👁️ Visual Characteristics (What the AI saw)", expanded=True):
                    st.write(f"**Condition:** {info['description']}")
                    st.markdown(f"**Typical Features:**\n{info['features']}")

                # 2. CAUSES (Specific)
                with st.expander("🧬 Etiology (Why this happened)"):
                    st.write(info['causes'])
                    
                # 3. TREATMENT (Specific)
                with st.expander("💊 Medical Treatment Options"):
                    st.info(info['treatment'])

                # --- ACTION PLAN (Visible in Dark Mode) ---
                st.markdown(f"""
                <div style='background-color: #f0f2f6; color: #000000; padding: 15px; border-radius: 10px; border-left: 5px solid #ff4b4b;'>
                    <strong>RECOMMENDED ACTION PLAN:</strong><br>
                    {info['action']}
                </div>
                """, unsafe_allow_html=True)
                
                st.divider()
                st.subheader("📊 Differential Diagnosis")
                chart_data = pd.DataFrame([
                    {"Condition": r['label'].replace('_', ' ').title(), "Probability (%)": r['score']*100} 
                    for r in st.session_state['results'][:3]
                ])
                st.bar_chart(chart_data.set_index("Condition"))
        else:
            st.info("Upload an image to begin diagnostic analysis.")

# --- TAB 2: DICTIONARY ---
with tab_dict:
    st.header("📚 Dermatological Encyclopedia")
    selected_cond = st.selectbox("Select Diagnosis:", list(MEDICAL_DB.keys()))
    data = MEDICAL_DB[selected_cond]
    
    st.subheader(f"📌 {selected_cond}")
    st.write(f"**Risk:** {data['risk_label']}")
    st.write(f"**Overview:** {data['description']}")
    st.markdown(f"**Signs:**\n{data['features']}")
    
    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("#### 🧬 Causes")
        st.write(data['causes'])
    with col_b:
        st.markdown("#### 💊 Treatment")
        st.write(data['treatment'])
        
    st.warning(f"**Directive:** {data['action']}")

# --- TAB 3: EMERGENCY ---
with tab_help:
    st.header("🚑 Specialist Locator")
    st.write("Locate the nearest Board-Certified Dermatologist.")
    if st.button("🔍 Find Dermatologist Near Me (Google Maps)"):
        webbrowser.open_new_tab("http://googleusercontent.com/maps.google.com/search?q=dermatologist+near+me")



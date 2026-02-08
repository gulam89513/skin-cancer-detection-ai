import streamlit as st
from transformers import pipeline
from PIL import Image
import pandas as pd
import webbrowser

# --- 1. PAGE CONFIGURATION ---
st.set_page_config(
    page_title="MediScan Pro | Advanced Dermatological Screening",
    page_icon="⚕️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- 2. ADVANCED MEDICAL DATABASE (Enhanced Accuracy & Details) ---
MEDICAL_DB = {
    "Actinic Keratoses": {
        "severity": "high",
        "risk_label": "PRE-CANCEROUS / HIGH RISK",
        "description": "A rough, scaly patch on the skin that develops from years of sun exposure. It is considered a precursor to skin cancer.",
        "symptoms": "Rough, sandpaper-like texture; red, pink, or brown patches; itching or burning sensation.",
        "causes": "☀️ **Root Cause:** Cumulative UV damage to DNA in skin cells from sunlight or tanning beds over many years.",
        "treatment": "💊 **Medical Cure:** Cryotherapy (freezing with liquid nitrogen), topical chemotherapy creams (5-fluorouracil), or Photodynamic Therapy (PDT).",
        "action": "⚠️ **URGENT:** Consult a dermatologist. If left untreated, 10-15% of these turn into Squamous Cell Carcinoma."
    },
    "Basal Cell Carcinoma": {
        "severity": "high",
        "risk_label": "MALIGNANT / HIGH RISK",
        "description": "The most common form of skin cancer. It grows slowly and rarely spreads (metastasizes), but can destroy local tissue.",
        "symptoms": "Pearly or waxy bump; flesh-colored or brown scar-like lesion; a bleeding or scabbing sore that heals and returns.",
        "causes": "☀️ **Root Cause:** Intense, intermittent sun exposure causing DNA mutations in basal cells.",
        "treatment": "💊 **Medical Cure:** Excision (cutting it out), Mohs Micrographic Surgery (highest cure rate), or Electrodessication and Curettage.",
        "action": "🚨 **ACTION:** Schedule a biopsy. Early removal results in a 100% cure rate. Do not ignore."
    },
    "Benign Keratosis": {
        "severity": "low",
        "risk_label": "BENIGN / HARMLESS",
        "description": "Also known as Seborrheic Keratosis. A non-cancerous skin growth that is very common in older adults.",
        "symptoms": "Waxy, stuck-on appearance; distinct raised edges; color ranges from tan to black.",
        "causes": "🧬 **Root Cause:** Genetic predisposition and aging. NOT caused by sun damage and NOT contagious.",
        "treatment": "💊 **Medical Cure:** No treatment required. Can be removed using Cryotherapy or Electrocautery for cosmetic reasons.",
        "action": "✅ **SAFE:** No action needed unless it becomes irritated or bleeds."
    },
    "Dermatofibroma": {
        "severity": "low",
        "risk_label": "BENIGN / HARMLESS",
        "description": "A common, non-cancerous skin growth that appears as a firm, localized nodule.",
        "symptoms": "Firm to the touch; dimples inward when pinched; usually pink or brown.",
        "causes": "🐜 **Root Cause:** Often an over-reaction to minor trauma, such as a bug bite, splinter, or shaving nick.",
        "treatment": "💊 **Medical Cure:** Harmless. Surgical excision involves leaving a scar, so it is usually left alone unless painful.",
        "action": "✅ **SAFE:** Monitor for size changes. No urgent care needed."
    },
    "Melanocytic Nevi": {
        "severity": "low",
        "risk_label": "BENIGN / MONITOR REQUIRED",
        "description": "Commonly known as a 'Mole'. A benign cluster of melanocytes (pigment-producing cells).",
        "symptoms": "Uniform color (brown/black); round or oval shape; distinct border.",
        "causes": "🧬 **Root Cause:** Genetic factors and sun exposure during childhood causes pigment cells to grow in clusters.",
        "treatment": "💊 **Medical Cure:** Surgical removal only if suspected of changing into melanoma or for cosmetic reasons.",
        "action": "🔍 **MONITOR:** Apply the 'ABCDE' Rule. If it changes in Asymmetry, Border, Color, Diameter, or Evolving, see a doctor."
    },
    "Melanoma": {
        "severity": "critical",
        "risk_label": "🔴 MALIGNANT / CRITICAL LIFE THREAT",
        "description": "The most dangerous form of skin cancer. It develops in the cells (melanocytes) that produce melanin.",
        "symptoms": "Asymmetrical shape; irregular borders; multiple colors (black, blue, red); diameter >6mm; evolving/changing over time.",
        "causes": "☀️ **Root Cause:** Unrepaired DNA damage to skin cells (from UV rays) triggers mutations that lead to rapid multiplication.",
        "treatment": "💊 **Medical Cure:** Wide Local Excision. Advanced stages require Immunotherapy, Targeted Therapy, or Radiation.",
        "action": "🚨 **EMERGENCY:** SEE A DOCTOR IMMEDIATELY. Early detection is the only way to ensure survival."
    },
    "Vascular Lesions": {
        "severity": "low",
        "description": "Abnormalities of blood vessels or other vessels. Includes cherry angiomas or spider veins.",
        "risk_label": "BENIGN / HARMLESS",
        "symptoms": "Bright red, purple, or blue spots; blanch (turn white) when pressed.",
        "causes": "🩸 **Root Cause:** Aging (Cherry Angiomas), hormonal changes (Pregnancy), or sun damage.",
        "treatment": "💊 **Medical Cure:** Laser therapy (Vascular Laser) is the gold standard for removal.",
        "action": "✅ **SAFE:** Usually harmless. Consult a doctor if the lesion bleeds extensively."
    }
}

# --- 3. MODEL LOADING ---
@st.cache_resource
def load_model():
    return pipeline("image-classification", model="Anwarkh1/Skin_Cancer-Image_Classification")

# --- 4. SIDEBAR (CONTROLS) ---
with st.sidebar:
    st.title("⚙️ MediScan Controls")
    st.divider()
    
    # Confidence Slider
    confidence_threshold = st.slider(
        "Accuracy Threshold (%)", 
        0, 100, 40, 
        help="Filters out low-quality or non-skin images. Higher settings require clearer photos."
    )
    
    st.divider()
    st.caption("Developed by Gulam")
    
    if st.button("🔄 Reset Analysis"):
        st.session_state.clear()
        st.rerun()

# --- 5. MAIN INTERFACE ---
st.title("🏥 MediScan Pro")
st.caption("Advanced AI Dermatological Screening System")

tab_scan, tab_dict, tab_help = st.tabs(["🔍 Clinical Scanner", "📚 Disease Encyclopedia", "🚑 Specialist Locator"])

# --- TAB 1: SCANNER ---
with tab_scan:
    col1, col2 = st.columns([0.8, 1.2])
    
    with col1:
        st.subheader("1. Specimen Input")
        st.info("📸 **Guidance:** Ensure the lesion is centered, well-lit, and in focus.")
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
            # Get Top Result
            top = st.session_state['results'][0]
            score = top['score'] * 100
            label_raw = top['label']
            label = label_raw.replace('_', ' ').title()
            
            # --- FILTER LOGIC ---
            if score < confidence_threshold:
                st.error("⚠️ ANALYSIS INCONCLUSIVE")
                st.warning(f"Confidence Level: {score:.1f}% (Below required {confidence_threshold}%)")
                st.markdown("""
                **Possible reasons for failure:**
                * Image is blurry or too dark.
                * Object is NOT a skin lesion (e.g., a book, face, or car).
                * The lesion is obscured by hair or shadow.
                """)
            else:
                # Retrieve Medical Data
                info = MEDICAL_DB.get(label, {
                    "severity": "low", 
                    "risk_label": "UNKNOWN",
                    "description": "Condition not found in database.", 
                    "symptoms": "Unknown",
                    "causes": "Unknown", 
                    "treatment": "Consult a doctor", 
                    "action": "Consult a doctor"
                })
                
                # --- DISPLAY PRIMARY RESULT ---
                if info['severity'] == "critical":
                    st.error(f"🔴 DETECTION: {label.upper()}")
                elif info['severity'] == "high":
                    st.warning(f"🟠 DETECTION: {label.upper()}")
                else:
                    st.success(f"🟢 DETECTION: {label.upper()}")

                st.write(f"**Risk Assessment:** {info['risk_label']}")
                st.metric("AI Confidence Probability", f"{score:.2f}%")
                
                st.divider()
                
                # --- DETAILED MEDICAL REPORT ---
                st.markdown("### 📋 Clinical Breakdown")
                
                with st.expander("📖 Description & Symptoms", expanded=True):
                    st.write(f"**What is it?** {info['description']}")
                    st.write(f"**Key Symptoms:** {info['symptoms']}")

                with st.expander("🧬 Etiology (Causes)"):
                    st.write(info['causes'])
                    
                with st.expander("💊 Medical Treatment & Cure"):
                    st.info(info['treatment'])

                # --- ACTION PLAN BOX (FIXED FOR DARK MODE) ---
                # Added 'color: black;' to ensure text is visible on the light gray background
                st.markdown(f"""
                <div style='background-color: #f0f2f6; color: #000000; padding: 15px; border-radius: 10px; border-left: 5px solid #ff4b4b;'>
                    <strong>RECOMMENDED ACTION PLAN:</strong><br>
                    {info['action']}
                </div>
                """, unsafe_allow_html=True)
                
                # --- DIFFERENTIAL DIAGNOSIS CHART ---
                st.divider()
                st.subheader("📊 Differential Diagnosis (Top 3)")
                st.caption("The AI considers these other possibilities:")
                
                # Format data for chart
                chart_data = pd.DataFrame([
                    {"Condition": r['label'].replace('_', ' ').title(), "Probability (%)": r['score']*100} 
                    for r in st.session_state['results'][:3] # Top 3 only
                ])
                st.bar_chart(chart_data.set_index("Condition"))
        else:
            st.info("Upload an image to begin diagnostic analysis.")

# --- TAB 2: DICTIONARY ---
with tab_dict:
    st.header("📚 Dermatological Encyclopedia")
    st.write("Comprehensive medical data on skin pathologies.")
    
    selected_cond = st.selectbox("Select Diagnosis:", list(MEDICAL_DB.keys()))
    data = MEDICAL_DB[selected_cond]
    
    st.subheader(f"📌 {selected_cond}")
    st.write(f"**Risk Level:** {data['risk_label']}")
    st.write(f"**Overview:** {data['description']}")
    
    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("#### 🧬 Causes")
        st.write(data['causes'])
    with col_b:
        st.markdown("#### 💊 Treatment")
        st.write(data['treatment'])
        
    st.warning(f"**Medical Directive:** {data['action']}")

# --- TAB 3: EMERGENCY ---
with tab_help:
    st.header("🚑 Specialist Locator")
    st.write("Locate the nearest Board-Certified Dermatologist.")
    
    if st.button("🔍 Find Dermatologist Near Me (Google Maps)"):
        webbrowser.open_new_tab("http://googleusercontent.com/maps.google.com/search?q=dermatologist+near+me")

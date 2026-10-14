import streamlit as st
import numpy as np
import pandas as pd
import pickle
import os
import sys

# ==========================================================
# NUMPY COMPATIBILITY SHIM (Cross NumPy 1.x & 2.x)
# ==========================================================
try:
    import numpy._core.multiarray
except ModuleNotFoundError:
    try:
        import numpy.core.multiarray
        sys.modules['numpy._core'] = sys.modules['numpy.core']
        sys.modules['numpy._core.multiarray'] = sys.modules['numpy.core.multiarray']
    except Exception:
        pass

# Import CatBoost explicitly
from catboost import CatBoostClassifier

# ==========================================================
# PAGE CONFIGURATION
# ==========================================================
st.set_page_config(
    page_title="Student Depression Screening System",
    page_icon="🧠",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# ==========================================================
# CUSTOM CSS STYLING
# ==========================================================
st.markdown("""
<style>
    .main {
        background-color: #F8FAFC;
    }
    .header-box {
        background: linear-gradient(135deg, #1E3A8A 0%, #2563EB 100%);
        padding: 2.2rem;
        border-radius: 14px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 4px 15px rgba(37, 99, 235, 0.2);
    }
    .header-box h1 {
        color: white !important;
        font-size: 28px;
        margin-bottom: 8px;
        font-weight: 700;
    }
    .header-box p {
        font-size: 15px;
        opacity: 0.92;
        margin-bottom: 0;
    }
    .card-depressed {
        background: linear-gradient(135deg, #FFF5F5 0%, #FED7D7 100%);
        border: 2px solid #E53E3E;
        border-left: 8px solid #E53E3E;
        padding: 1.8rem;
        border-radius: 12px;
        margin-top: 1.5rem;
        margin-bottom: 1.5rem;
        box-shadow: 0 4px 12px rgba(229, 62, 62, 0.15);
    }
    .card-not-depressed {
        background: linear-gradient(135deg, #F0FFF4 0%, #C6F6D5 100%);
        border: 2px solid #38A169;
        border-left: 8px solid #38A169;
        padding: 1.8rem;
        border-radius: 12px;
        margin-top: 1.5rem;
        margin-bottom: 1.5rem;
        box-shadow: 0 4px 12px rgba(56, 161, 105, 0.15);
    }
    .result-title {
        font-size: 26px;
        font-weight: 800;
        margin-bottom: 10px;
    }
    .stButton>button {
        width: 100%;
        height: 52px;
        font-size: 18px;
        font-weight: 700;
        background: linear-gradient(135deg, #2563EB 0%, #1D4ED8 100%);
        color: white;
        border: none;
        border-radius: 10px;
        transition: all 0.3s ease;
        box-shadow: 0 4px 10px rgba(37, 99, 235, 0.25);
    }
    .stButton>button:hover {
        background: linear-gradient(135deg, #1D4ED8 0%, #1E40AF 100%);
        box-shadow: 0 6px 16px rgba(37, 99, 235, 0.4);
    }
</style>
""", unsafe_allow_html=True)

# ==========================================================
# BACKEND: MODEL LOADING & INFERENCE
# ==========================================================
@st.cache_resource
def load_backend_model():
    # Priority 1: CatBoost native CBM format (100% portable, zero pickle/numpy version conflicts)
    cbm_candidates = [
        'catboost_model.cbm',
        'new dataset/catboost_model.cbm',
        './catboost_model.cbm',
        './new dataset/catboost_model.cbm'
    ]
    for cbm_path in cbm_candidates:
        if os.path.exists(cbm_path):
            cb_model = CatBoostClassifier()
            cb_model.load_model(cbm_path)
            return cb_model

    # Priority 2: Pickle bundle format (.sav)
    sav_candidates = [
        'new dataset/Quantum_CatBoost_Model_Qiskit.sav',
        './new dataset/Quantum_CatBoost_Model_Qiskit.sav',
        'Quantum_CatBoost_Model_Qiskit.sav',
        './Quantum_CatBoost_Model_Qiskit.sav'
    ]
    for sav_path in sav_candidates:
        if os.path.exists(sav_path):
            with open(sav_path, 'rb') as f:
                bundle = pickle.load(f)
            if isinstance(bundle, dict) and 'model' in bundle:
                return bundle['model']
            return bundle
            
    st.error("❌ Model file not found. Please ensure 'catboost_model.cbm' or 'Quantum_CatBoost_Model_Qiskit.sav' is uploaded.")
    st.stop()

model = load_backend_model()

def run_backend_prediction(features_list):
    """
    All quantum feature encoding runs strictly in the backend:
    Angle embedding mapping x -> cos(x) Pauli-Z expectation values
    """
    raw_vector = np.array(features_list, dtype=np.float32).reshape(1, -1)
    # Quantum feature vector transformation in backend
    quantum_features = np.cos(raw_vector)
    
    # Model inference
    pred_label = int(model.predict(quantum_features).ravel()[0])
    pred_prob = model.predict_proba(quantum_features)[0]
    
    return pred_label, pred_prob

# ==========================================================
# USER INTERFACE (ALL IN ENGLISH)
# ==========================================================
st.markdown("""
<div class="header-box">
    <h1>🧠 Student Depression Screening System</h1>
    <p>Automated Clinical Screening Platform for Student Psychological Health Assessment</p>
</div>
""", unsafe_allow_html=True)

st.write("Please provide the 8 required parameters below and click **'Check Depression Status'**:")

col1, col2 = st.columns(2)

with col1:
    st.markdown("##### 🩺 Personal & Physical Health")
    
    # 1. Health Condition
    health_opts = {'Good': 2, 'Normal': 3, 'Abnormal': 0, 'Critical': 1}
    health_sel = st.selectbox(
        "1. Physical Health Condition", 
        options=list(health_opts.keys()), 
        index=0,
        help="General physiological wellbeing rating."
    )
    val_health = health_opts[health_sel]
    
    # 2. Relationship Status
    rel_opts = {'Single': 2, 'In a Relationship': 1, 'Complicated': 0}
    rel_sel = st.selectbox(
        "2. Relationship Status",
        options=list(rel_opts.keys()),
        index=0,
        help="Current interpersonal status."
    )
    val_rel = rel_opts[rel_sel]
    
    # 3. Living Style
    living_opts = {'Family (Living with Parents)': 1, 'Bachelor (Hostel / Mess)': 0}
    living_sel = st.selectbox(
        "3. Living Arrangement",
        options=list(living_opts.keys()),
        index=0,
        help="Living with family vs independent bachelor accommodation."
    )
    val_living = living_opts[living_sel]
    
    # 4. Marital Status
    marital_opts = {'Unmarried': 2, 'Married': 0, 'Separated': 1}
    marital_sel = st.selectbox(
        "4. Marital Status",
        options=list(marital_opts.keys()),
        index=0
    )
    val_marital = marital_opts[marital_sel]

with col2:
    st.markdown("##### 📚 Academic & Socioeconomic Context")
    
    # 5. CGPA
    val_cgpa = st.number_input(
        "5. Cumulative GPA (CGPA: 2.00 - 4.00)",
        min_value=2.00,
        max_value=4.00,
        value=3.40,
        step=0.01,
        help="Current university CGPA on a 4.00 scale."
    )
    
    # 6. Cost without Study (BDT)
    val_cost = st.number_input(
        "6. Monthly Out-of-Pocket Living Expenses (BDT)",
        min_value=0,
        max_value=35000,
        value=11000,
        step=500,
        help="Baseline monthly living costs incurred independent of tuition fees."
    )
    
    # 7. Friend Circle Support
    friend_opts = {'Yes (Active Social Circle)': 1, 'No (Socially Isolated)': 0}
    friend_sel = st.selectbox(
        "7. Peer Support / Friend Circle",
        options=list(friend_opts.keys()),
        index=0,
        help="Presence of close friends and emotional social support."
    )
    val_friend = friend_opts[friend_sel]
    
    # 8. Family Background
    fam_opts = {'Middle class': 1, 'Lower Middle class': 0, 'Rich class': 2}
    fam_sel = st.selectbox(
        "8. Family Socioeconomic Background",
        options=list(fam_opts.keys()),
        index=0,
        help="Family socioeconomic tier."
    )
    val_fam = fam_opts[fam_sel]

st.markdown("<br>", unsafe_allow_html=True)
submit_btn = st.button("🔍 Check Depression Status")

# ==========================================================
# PREDICTION & CLEAN RESULT DISPLAY
# ==========================================================
if submit_btn:
    # Feature vector ordered exactly as expected by the model
    features = [
        val_health,
        val_rel,
        float(val_cost),
        float(val_cgpa),
        val_living,
        val_marital,
        val_friend,
        val_fam
    ]
    
    # Run prediction in backend (quantum encoding kept completely internal)
    pred_label, pred_prob = run_backend_prediction(features)
    prob_depressed = pred_prob[1] * 100
    prob_not_depressed = pred_prob[0] * 100
    
    if pred_label == 1:
        st.markdown(f"""
        <div class="card-depressed">
            <div class="result-title" style="color:#C53030;">
                ⚠️ Depression: Yes
            </div>
            <p style="font-size:18px; font-weight:600; color:#9B2C2C; margin-bottom:8px;">
                Target Prediction: <span style="background:#FED7D7; padding:4px 10px; border-radius:6px; color:#C53030;">Depressed (Yes)</span>
            </p>
            <p style="font-size:16px; color:#4A1515; margin-bottom:12px;">
                The student is predicted to have depression with <b>{prob_depressed:.1f}% confidence</b>.
            </p>
            <div style="background:white; padding:14px 20px; border-radius:8px; border:1px solid #FEB2B2;">
                <h5 style="color:#9B2C2C; margin-top:0;">💡 Recommended Clinical Steps:</h5>
                <ul style="color:#742A2A; margin-bottom:0; font-size:15px;">
                    <li><b>Urgent Consultation:</b> Schedule an appointment with a licensed psychologist or university counseling center.</li>
                    <li><b>Social Support:</b> Open up and share your feelings with trusted family members or close friends.</li>
                    <li><b>Self-Care:</b> Prioritize regular sleep hygiene, light physical exercise, and stress-reduction routines.</li>
                </ul>
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="card-not-depressed">
            <div class="result-title" style="color:#276749;">
                ✅ Depression: No
            </div>
            <p style="font-size:18px; font-weight:600; color:#22543D; margin-bottom:8px;">
                Target Prediction: <span style="background:#C6F6D5; padding:4px 10px; border-radius:6px; color:#22543D;">Not Depressed (No)</span>
            </p>
            <p style="font-size:16px; color:#1C4532; margin-bottom:12px;">
                The student is predicted to have no depression with <b>{prob_not_depressed:.1f}% confidence</b>.
            </p>
            <div style="background:white; padding:14px 20px; border-radius:8px; border:1px solid #9AE6B4;">
                <h5 style="color:#22543D; margin-top:0;">🌱 Preventive Wellness Tips:</h5>
                <ul style="color:#22543D; margin-bottom:0; font-size:15px;">
                    <li><b>Maintain Healthy Habits:</b> Keep up regular sleep patterns, physical activity, and balanced nutrition.</li>
                    <li><b>Balanced Routine:</b> Engage in extracurricular activities, social interactions, and creative hobbies.</li>
                    <li><b>Open Communication:</b> Never hesitate to reach out for guidance if you experience elevated academic or personal stress.</li>
                </ul>
            </div>
        </div>
        """, unsafe_allow_html=True)

    # Clean progress bar
    st.write(f"**Depression Probability Score:** `{prob_depressed:.1f}%`")
    st.progress(float(prob_depressed) / 100.0)

st.markdown("<br><hr>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center; color:#94A3B8; font-size:13px;'>Student Mental Health Screening System | Clinical Screening Aid</p>", unsafe_allow_html=True)

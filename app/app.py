import streamlit as st
import joblib
import numpy as np
import pandas as pd
import shap
import time
import io
from fpdf import FPDF

# Page config
st.set_page_config(
    page_title="GlucoseGuard PK",
    page_icon="🩺",
    layout="wide"
)

# CSS
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

* { font-family: 'Inter', sans-serif; }

.stApp { background-color: #f8fafc; }

.top-banner {
    background: linear-gradient(135deg, #0d9488 0%, #0f766e 50%, #134e4a 100%);
    padding: 3rem;
    border-radius: 24px;
    margin-bottom: 2rem;
    box-shadow: 0 20px 60px rgba(13,148,136,0.4);
    position: relative;
    overflow: hidden;
}
.top-banner::before {
    content: '';
    position: absolute;
    top: -50%;
    right: -10%;
    width: 400px;
    height: 400px;
    background: rgba(255,255,255,0.05);
    border-radius: 50%;
}
.top-banner::after {
    content: '';
    position: absolute;
    bottom: -30%;
    right: 20%;
    width: 200px;
    height: 200px;
    background: rgba(255,255,255,0.05);
    border-radius: 50%;
}
.banner-title {
    color: white;
    font-size: 3rem;
    font-weight: 900;
    margin: 0;
    letter-spacing: -1px;
    text-shadow: 0 2px 10px rgba(0,0,0,0.2);
}
.banner-sub {
    color: rgba(255,255,255,0.9);
    font-size: 1.15rem;
    margin: 0.5rem 0 1.5rem 0;
}
.banner-badge {
    display: inline-block;
    background: rgba(255,255,255,0.15);
    backdrop-filter: blur(10px);
    border: 1px solid rgba(255,255,255,0.3);
    color: white;
    padding: 0.5rem 1.5rem;
    border-radius: 50px;
    font-size: 0.9rem;
    font-weight: 600;
    margin-right: 0.5rem;
}

.stat-card {
    background: white;
    border-radius: 20px;
    padding: 1.5rem;
    text-align: center;
    box-shadow: 0 4px 20px rgba(0,0,0,0.08);
    border-top: 4px solid #0d9488;
    transition: transform 0.2s;
}
.stat-card:hover { transform: translateY(-4px); }
.stat-number {
    font-size: 2.2rem;
    font-weight: 900;
    color: #0d9488;
    margin: 0;
}
.stat-label {
    color: #6b7280;
    font-size: 0.85rem;
    margin: 0.3rem 0 0 0;
    font-weight: 500;
}

.section-card {
    background: white;
    border-radius: 20px;
    padding: 2rem;
    box-shadow: 0 4px 20px rgba(0,0,0,0.06);
    margin-bottom: 1.5rem;
    border: 1px solid #f0fdf4;
}
.section-title {
    color: #0f766e;
    font-size: 0.85rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 2px;
    margin-bottom: 1.5rem;
    padding-bottom: 0.8rem;
    border-bottom: 2px solid #ccfbf1;
}

.result-low {
    background: linear-gradient(135deg, #ecfdf5, #d1fae5);
    border: 2px solid #10b981;
    border-radius: 24px;
    padding: 2.5rem;
    text-align: center;
    box-shadow: 0 8px 30px rgba(16,185,129,0.2);
}
.result-medium {
    background: linear-gradient(135deg, #fffbeb, #fef3c7);
    border: 2px solid #f59e0b;
    border-radius: 24px;
    padding: 2.5rem;
    text-align: center;
    box-shadow: 0 8px 30px rgba(245,158,11,0.2);
}
.result-high {
    background: linear-gradient(135deg, #fff1f2, #fee2e2);
    border: 2px solid #ef4444;
    border-radius: 24px;
    padding: 2.5rem;
    text-align: center;
    box-shadow: 0 8px 30px rgba(239,68,68,0.2);
}
.result-score {
    font-size: 5rem;
    font-weight: 900;
    margin: 0;
    line-height: 1;
}
.result-label {
    font-size: 1.5rem;
    font-weight: 800;
    margin: 0.5rem 0;
}
.action-box {
    background: white;
    border-radius: 14px;
    padding: 1rem 1.5rem;
    margin-top: 1rem;
    font-weight: 600;
    font-size: 0.95rem;
}

.driver-card {
    background: #f8fafc;
    border-radius: 14px;
    padding: 1rem 1.2rem;
    margin: 0.5rem 0;
    border-left: 5px solid #0d9488;
    display: flex;
    align-items: center;
    gap: 1rem;
}

.info-pill {
    background: #f0fdf4;
    border: 1px solid #bbf7d0;
    color: #065f46;
    padding: 0.3rem 0.8rem;
    border-radius: 50px;
    font-size: 0.8rem;
    font-weight: 600;
    display: inline-block;
    margin: 0.2rem;
}

.patient-row {
    background: white;
    border-radius: 12px;
    padding: 1rem;
    margin: 0.5rem 0;
    border-left: 5px solid #0d9488;
    box-shadow: 0 2px 8px rgba(0,0,0,0.05);
}

.pakistan-stat {
    background: linear-gradient(135deg, #0d9488, #0f766e);
    border-radius: 16px;
    padding: 1.5rem;
    text-align: center;
    color: white;
}
.pakistan-stat h2 {
    font-size: 2.5rem;
    font-weight: 900;
    margin: 0;
}
.pakistan-stat p {
    margin: 0.3rem 0 0 0;
    opacity: 0.9;
    font-size: 0.9rem;
}

.stButton > button {
    background: linear-gradient(135deg, #0d9488, #0f766e) !important;
    color: white !important;
    border: none !important;
    border-radius: 14px !important;
    padding: 0.8rem 2rem !important;
    font-size: 1.05rem !important;
    font-weight: 700 !important;
    width: 100% !important;
    box-shadow: 0 4px 20px rgba(13,148,136,0.4) !important;
}

div[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #134e4a, #0f766e) !important;
}
div[data-testid="stSidebar"] * {
    color: white !important;
}
div[data-testid="stSidebar"] .stRadio label {
    color: white !important;
}

.footer {
    background: linear-gradient(135deg, #134e4a, #0f766e);
    color: rgba(255,255,255,0.8);
    text-align: center;
    padding: 2rem;
    border-radius: 20px;
    margin-top: 2rem;
}
</style>
""", unsafe_allow_html=True)

# Load model
model = joblib.load('models/glucoseguard_model.pkl')
explainer = joblib.load('models/shap_explainer.pkl')

# Session state for camp tracker
if 'patients' not in st.session_state:
    st.session_state.patients = []

# Feature labels
LABELS = {
    'HighBP': 'High Blood Pressure',
    'HighChol': 'High Cholesterol',
    'CholCheck': 'Cholesterol Check',
    'BMI': 'Body Mass Index',
    'Smoker': 'Smoking',
    'Stroke': 'Stroke History',
    'HeartDiseaseorAttack': 'Heart Disease',
    'PhysActivity': 'Physical Activity',
    'Fruits': 'Fruit Intake',
    'Veggies': 'Vegetable Intake',
    'HvyAlcoholConsump': 'Alcohol Use',
    'AnyHealthcare': 'Healthcare Access',
    'NoDocbcCost': 'Avoided Doctor (Cost)',
    'GenHlth': 'General Health',
    'MentHlth': 'Mental Health Days',
    'PhysHlth': 'Physical Health Days',
    'DiffWalk': 'Difficulty Walking',
    'Sex': 'Sex',
    'Age': 'Age Group',
    'Education': 'Education',
    'Income': 'Income'
}

AGE_MAP = {
    1:"18-24", 2:"25-29", 3:"30-34", 4:"35-39",
    5:"40-44", 6:"45-49", 7:"50-54", 8:"55-59",
    9:"60-64", 10:"65-69", 11:"70-74", 12:"75-79", 13:"80+"
}

# Banner
st.markdown("""
<div class="top-banner">
    <p class="banner-title">🩺 GlucoseGuard PK</p>
    <p class="banner-sub">AI-Assisted Diabetes Triage & Referral Support System</p>
    <span class="banner-badge">🇵🇰 Built for Pakistan</span>
    <span class="banner-badge">🏥 Community Health Workers</span>
    <span class="banner-badge">🤖 AI Powered</span>
</div>
""", unsafe_allow_html=True)

# Stats row
c1, c2, c3, c4 = st.columns(4)
with c1:
    st.markdown("""<div class="stat-card">
        <p class="stat-number">33M+</p>
        <p class="stat-label">Diabetics in Pakistan</p>
    </div>""", unsafe_allow_html=True)
with c2:
    st.markdown("""<div class="stat-card">
        <p class="stat-number">50%</p>
        <p class="stat-label">Cases Undiagnosed</p>
    </div>""", unsafe_allow_html=True)
with c3:
    st.markdown("""<div class="stat-card">
        <p class="stat-number">3rd</p>
        <p class="stat-label">Highest in the World</p>
    </div>""", unsafe_allow_html=True)
with c4:
    st.markdown("""<div class="stat-card">
        <p class="stat-number">80K+</p>
        <p class="stat-label">Records Used to Train AI</p>
    </div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# Sidebar
st.sidebar.markdown("## 🩺 GlucoseGuard PK")
st.sidebar.markdown("---")
mode = st.sidebar.radio("Navigate", [
    "🔍 Single Patient Screening",
    "👥 Camp Session Tracker",
    "📊 Population Upload",
    "🇵🇰 Pakistan Diabetes Info"
])
st.sidebar.markdown("---")
st.sidebar.markdown("""
**Risk Guide**

🟢 Under 30% — Low Risk
Lifestyle advice only

🟡 30 to 60% — Medium Risk
Retest within 1 month

🔴 Above 60% — High Risk
Refer for HbA1c immediately
""")
st.sidebar.markdown("---")
st.sidebar.caption("⚠️ For screening support only. Does not replace clinical diagnosis.")

# Helper: get prediction
def get_prediction(inputs):
    input_df = pd.DataFrame([inputs])
    prob = model.predict_proba(input_df)[0][1]
    return round(prob * 100, 1)

def get_risk_info(risk_pct):
    if risk_pct < 30:
        return "LOW", "result-low", "🟢", "✅ Provide lifestyle counselling. Rescreen in 12 months."
    elif risk_pct < 60:
        return "MEDIUM", "result-medium", "🟡", "⚠️ Repeat fasting glucose test within 1 month. Dietary advice recommended."
    else:
        return "HIGH", "result-high", "🔴", "🚨 Refer for HbA1c testing within 2 weeks. Immediate clinical attention required."

def get_shap_drivers(input_df):
    shap_vals = explainer.shap_values(input_df)
    df = pd.DataFrame({
        'Feature': input_df.columns,
        'Impact': shap_vals[0],
        'Abs': abs(shap_vals[0])
    }).nlargest(5, 'Abs')
    return df

def build_inputs(age, sex, bmi, highbp, highchol, cholcheck,
                 heartdisease, stroke, smoker, physactivity,
                 fruits, veggies, hvyalcohol, anyhealthcare,
                 nodocbccost, genhlth, menthlth, physhlth, diffwalk):
    return {
        'HighBP': highbp, 'HighChol': highchol, 'CholCheck': cholcheck,
        'BMI': bmi, 'Smoker': smoker, 'Stroke': stroke,
        'HeartDiseaseorAttack': heartdisease, 'PhysActivity': physactivity,
        'Fruits': fruits, 'Veggies': veggies, 'HvyAlcoholConsump': hvyalcohol,
        'AnyHealthcare': anyhealthcare, 'NoDocbcCost': nodocbccost,
        'GenHlth': genhlth, 'MentHlth': menthlth, 'PhysHlth': physhlth,
        'DiffWalk': diffwalk, 'Sex': sex, 'Age': age,
        'Education': 4, 'Income': 5
    }

def patient_form(prefix=""):
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("**🔢 Demographics**")
        age = st.select_slider(f"Age Group{prefix}",
            options=list(AGE_MAP.keys()),
            format_func=lambda x: AGE_MAP[x], value=5)
        sex = st.selectbox(f"Sex{prefix}", [0,1],
            format_func=lambda x: "Male" if x==1 else "Female")
        
        st.markdown("**📏 BMI Calculator**")
        height_cm = st.number_input(f"Height (cm){prefix}", 100, 220, 165)
        weight_kg = st.number_input(f"Weight (kg){prefix}", 30, 200, 70)
        bmi = round(weight_kg / ((height_cm/100)**2), 1)
        st.markdown(f"""<div class="info-pill">📊 BMI: {bmi}</div>""",
            unsafe_allow_html=True)

    with col2:
        st.markdown("**🩸 Clinical Indicators**")
        highbp = st.selectbox(f"High Blood Pressure?{prefix}", [0,1],
            format_func=lambda x: "Yes" if x==1 else "No")
        highchol = st.selectbox(f"High Cholesterol?{prefix}", [0,1],
            format_func=lambda x: "Yes" if x==1 else "No")
        cholcheck = st.selectbox(f"Cholesterol Checked (5 yrs)?{prefix}", [0,1],
            format_func=lambda x: "Yes" if x==1 else "No")
        heartdisease = st.selectbox(f"Heart Disease History?{prefix}", [0,1],
            format_func=lambda x: "Yes" if x==1 else "No")
        stroke = st.selectbox(f"Stroke History?{prefix}", [0,1],
            format_func=lambda x: "Yes" if x==1 else "No")
        genhlth = st.select_slider(f"General Health{prefix}",
            options=[1,2,3,4,5],
            format_func=lambda x: {1:"Excellent",2:"Very Good",
                3:"Good",4:"Fair",5:"Poor"}[x], value=3)

    with col3:
        st.markdown("**🏃 Lifestyle**")
        smoker = st.selectbox(f"Smoker?{prefix}", [0,1],
            format_func=lambda x: "Yes" if x==1 else "No")
        physactivity = st.selectbox(f"Physically Active?{prefix}", [0,1],
            format_func=lambda x: "Yes" if x==1 else "No")
        fruits = st.selectbox(f"Eats Fruits Daily?{prefix}", [0,1],
            format_func=lambda x: "Yes" if x==1 else "No")
        veggies = st.selectbox(f"Eats Vegetables Daily?{prefix}", [0,1],
            format_func=lambda x: "Yes" if x==1 else "No")
        hvyalcohol = st.selectbox(f"Heavy Alcohol Use?{prefix}", [0,1],
            format_func=lambda x: "Yes" if x==1 else "No")
        diffwalk = st.selectbox(f"Difficulty Walking?{prefix}", [0,1],
            format_func=lambda x: "Yes" if x==1 else "No")
        anyhealthcare = st.selectbox(f"Has Healthcare Access?{prefix}", [0,1],
            format_func=lambda x: "Yes" if x==1 else "No")
        nodocbccost = st.selectbox(f"Avoided Doctor (Cost)?{prefix}", [0,1],
            format_func=lambda x: "Yes" if x==1 else "No")

    st.markdown("---")
    col4, col5 = st.columns(2)
    with col4:
        menthlth = st.slider(f"Poor Mental Health Days (last 30){prefix}", 0, 30, 0)
    with col5:
        physhlth = st.slider(f"Poor Physical Health Days (last 30){prefix}", 0, 30, 0)

    return build_inputs(age, sex, bmi, highbp, highchol, cholcheck,
                        heartdisease, stroke, smoker, physactivity,
                        fruits, veggies, hvyalcohol, anyhealthcare,
                        nodocbccost, genhlth, menthlth, physhlth, diffwalk)

def generate_pdf(name, risk_pct, risk_level, action, drivers):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", "B", 20)
    pdf.set_text_color(13, 148, 136)
    pdf.cell(0, 12, "GlucoseGuard PK - Patient Report", ln=True)
    pdf.set_font("Arial", "", 11)
    pdf.set_text_color(100, 100, 100)
    pdf.cell(0, 8, "AI-Assisted Diabetes Triage & Referral Support System", ln=True)
    pdf.ln(5)
    pdf.set_draw_color(13, 148, 136)
    pdf.line(10, pdf.get_y(), 200, pdf.get_y())
    pdf.ln(5)
    pdf.set_font("Arial", "B", 13)
    pdf.set_text_color(30, 30, 30)
    pdf.cell(0, 10, f"Patient: {name}", ln=True)
    pdf.set_font("Arial", "", 11)
    pdf.set_text_color(80, 80, 80)
    pdf.cell(0, 8, f"Date: {pd.Timestamp.now().strftime('%d %B %Y')}", ln=True)
    pdf.ln(5)
    pdf.set_font("Arial", "B", 14)
    pdf.set_text_color(13, 148, 136)
    pdf.cell(0, 10, f"Risk Score: {risk_pct}%  |  Risk Level: {risk_level}", ln=True)
    pdf.ln(3)
    pdf.set_font("Arial", "B", 12)
    pdf.set_text_color(30, 30, 30)
    pdf.cell(0, 8, "Recommended Action:", ln=True)
    pdf.set_font("Arial", "", 11)
    pdf.set_text_color(80, 80, 80)
    clean_action = action.replace("✅","").replace("⚠️","").replace("🚨","").strip()
    pdf.multi_cell(0, 8, clean_action)
    pdf.ln(5)
    pdf.set_font("Arial", "B", 12)
    pdf.set_text_color(30, 30, 30)
    pdf.cell(0, 8, "Key Risk Drivers:", ln=True)
    pdf.set_font("Arial", "", 11)
    for _, row in drivers.iterrows():
        direction = "Increases risk" if row['Impact'] > 0 else "Decreases risk"
        name_label = LABELS.get(row['Feature'], row['Feature'])
        pdf.cell(0, 8, f"  - {name_label}: {direction}", ln=True)
    pdf.ln(8)
    pdf.set_font("Arial", "I", 9)
    pdf.set_text_color(150, 150, 150)
    pdf.multi_cell(0, 6, "DISCLAIMER: This report is generated by an AI screening tool for triage support only. It does not constitute a clinical diagnosis. Please refer high-risk patients to a qualified healthcare professional.")
    return pdf.output(dest='S').encode('latin-1')

# ============ PAGES ============

if "Single" in mode:
    st.markdown('<div class="section-card"><div class="section-title">👤 Single Patient Screening</div>', unsafe_allow_html=True)
    patient_name = st.text_input("Patient Name or ID", placeholder="e.g. Patient 001 or Ali Hassan")
    inputs = patient_form(prefix="")
    st.markdown("</div>", unsafe_allow_html=True)

    if st.button("🔍 Assess Diabetes Risk", type="primary"):
        with st.spinner("🧠 Analyzing patient profile..."):
            time.sleep(1.2)

        risk_pct = get_prediction(inputs)
        risk_level, css_class, emoji, action = get_risk_info(risk_pct)
        input_df = pd.DataFrame([inputs])
        drivers = get_shap_drivers(input_df)

        st.markdown("---")
        st.markdown("## 📊 Assessment Results")

        col_a, col_b = st.columns(2)

        with col_a:
            progress = st.progress(0)
            for i in range(int(risk_pct)):
                time.sleep(0.012)
                progress.progress(i+1)

            st.markdown(f"""
            <div class="{css_class}">
                <div class="result-score">{risk_pct}%</div>
                <div class="result-label">{emoji} {risk_level} RISK</div>
                <div class="action-box">{action}</div>
            </div>
            """, unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)

            pdf_bytes = generate_pdf(
                patient_name or "Unknown", risk_pct,
                risk_level, action, drivers
            )
            st.download_button(
                label="📄 Download Patient Report (PDF)",
                data=pdf_bytes,
                file_name=f"GlucoseGuard_{patient_name or 'Patient'}_Report.pdf",
                mime="application/pdf"
            )

        with col_b:
            st.markdown("### 🔍 Key Risk Drivers")
            st.caption("Factors most influencing this patient's risk score:")
            for _, row in drivers.iterrows():
                direction = "🔺 Increases risk" if row['Impact'] > 0 else "🔻 Decreases risk"
                color = "#ef4444" if row['Impact'] > 0 else "#10b981"
                label = LABELS.get(row['Feature'], row['Feature'])
                st.markdown(f"""
                <div class="driver-card" style="border-left-color:{color}">
                    <div>
                        <b style="color:#1f2937;font-size:1rem">{label}</b><br>
                        <span style="color:{color};font-size:0.85rem">{direction}</span>
                    </div>
                </div>
                """, unsafe_allow_html=True)

        if st.button("➕ Add to Camp Session"):
            st.session_state.patients.append({
                'Name': patient_name or f"Patient {len(st.session_state.patients)+1}",
                'Risk Score': risk_pct,
                'Risk Level': risk_level,
                'Action': action.replace("✅","").replace("⚠️","").replace("🚨","").strip()
            })
            st.success(f"✅ Added {patient_name or 'Patient'} to camp session!")

elif "Camp" in mode:
    st.markdown('<div class="section-card"><div class="section-title">👥 Camp Session Tracker</div>', unsafe_allow_html=True)
    st.markdown("Screen multiple patients during a camp session. All results are ranked by risk — highest risk first.")
    st.markdown("</div>", unsafe_allow_html=True)

    if len(st.session_state.patients) == 0:
        st.info("No patients screened yet. Go to Single Patient Screening and click 'Add to Camp Session' after each assessment.")
    else:
        patients_df = pd.DataFrame(st.session_state.patients)
        patients_df = patients_df.sort_values('Risk Score', ascending=False).reset_index(drop=True)

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("👥 Total Screened", len(patients_df))
        col2.metric("🔴 High Risk", len(patients_df[patients_df['Risk Level']=='HIGH']))
        col3.metric("🟡 Medium Risk", len(patients_df[patients_df['Risk Level']=='MEDIUM']))
        col4.metric("🟢 Low Risk", len(patients_df[patients_df['Risk Level']=='LOW']))

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("### 📋 Patient Priority List (Highest Risk First)")

        for i, row in patients_df.iterrows():
            if row['Risk Level'] == 'HIGH':
                color = "#ef4444"
                emoji = "🔴"
            elif row['Risk Level'] == 'MEDIUM':
                color = "#f59e0b"
                emoji = "🟡"
            else:
                color = "#10b981"
                emoji = "🟢"

            st.markdown(f"""
            <div class="patient-row" style="border-left-color:{color}">
                <b style="font-size:1.05rem">{i+1}. {row['Name']}</b>
                <span style="float:right;font-weight:800;color:{color};font-size:1.1rem">
                    {emoji} {row['Risk Score']}% — {row['Risk Level']}
                </span><br>
                <span style="color:#6b7280;font-size:0.9rem">{row['Action']}</span>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        csv = patients_df.to_csv(index=False)
        st.download_button(
            "📥 Download Session Report (CSV)",
            data=csv,
            file_name="GlucoseGuard_Camp_Session.csv",
            mime="text/csv"
        )

        if st.button("🗑️ Clear Session"):
            st.session_state.patients = []
            st.success("Session cleared!")

elif "Population" in mode:
    st.markdown('<div class="section-card"><div class="section-title">📊 Population Upload & Analysis</div>', unsafe_allow_html=True)
    st.markdown("Upload a CSV file of multiple patients to get instant population-level risk insights.")
    st.markdown("</div>", unsafe_allow_html=True)

    uploaded_file = st.file_uploader("Upload Patient CSV", type=['csv'])
    if uploaded_file:
        with st.spinner("Analyzing population..."):
            pop_df = pd.read_csv(uploaded_file)
            probs = model.predict_proba(pop_df)[:,1] * 100
            time.sleep(1)

        low = sum(probs < 30)
        medium = sum((probs >= 30) & (probs < 60))
        high = sum(probs >= 60)
        total = len(probs)

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("👥 Total", total)
        col2.metric("🟢 Low Risk", f"{low} ({round(low/total*100)}%)")
        col3.metric("🟡 Medium Risk", f"{medium} ({round(medium/total*100)}%)")
        col4.metric("🔴 High Risk", f"{high} ({round(high/total*100)}%)")

        st.markdown("### 📈 Risk Distribution")
        risk_dist = pd.DataFrame({
            'Risk Level': ['Low Risk', 'Medium Risk', 'High Risk'],
            'Count': [low, medium, high]
        })
        st.bar_chart(risk_dist.set_index('Risk Level'))

        st.markdown("### 📈 Key Risk Factors in Population")
        risk_cols = ['HighBP','HighChol','BMI','Smoker','PhysActivity']
        available = [c for c in risk_cols if c in pop_df.columns]
        if available:
            st.bar_chart(pop_df[available].mean())

elif "Pakistan" in mode:
    st.markdown('<div class="section-card"><div class="section-title">🇵🇰 Pakistan Diabetes Crisis</div>', unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("""<div class="pakistan-stat">
            <h2>33M+</h2><p>People with diabetes in Pakistan</p>
        </div>""", unsafe_allow_html=True)
    with c2:
        st.markdown("""<div class="pakistan-stat">
            <h2>3rd</h2><p>Highest diabetic population globally</p>
        </div>""", unsafe_allow_html=True)
    with c3:
        st.markdown("""<div class="pakistan-stat">
            <h2>50%</h2><p>Cases that remain undiagnosed</p>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    st.markdown("""
    <div class="section-card">
        <div class="section-title">📍 Highest Burden Regions</div>
        <p>Diabetes prevalence is highest in:</p>
        <span class="info-pill">🏙️ Karachi</span>
        <span class="info-pill">🏙️ Lahore</span>
        <span class="info-pill">🏙️ Islamabad</span>
        <span class="info-pill">🌾 Rural Sindh</span>
        <span class="info-pill">🌾 Southern Punjab</span>
        <br><br>
        <p style="color:#6b7280">Urban areas show higher prevalence due to sedentary lifestyles, 
        processed food consumption, and rising obesity rates. Rural areas face the challenge 
        of low awareness and limited access to screening facilities.</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="section-card">
        <div class="section-title">⚠️ Key Risk Factors in Pakistan</div>
        <span class="info-pill">🍚 High Carbohydrate Diet</span>
        <span class="info-pill">🪑 Sedentary Lifestyle</span>
        <span class="info-pill">🧬 Genetic Predisposition</span>
        <span class="info-pill">⚖️ Rising Obesity</span>
        <span class="info-pill">💊 Low Awareness</span>
        <span class="info-pill">🏥 Limited Healthcare Access</span>
        <span class="info-pill">🚬 High Smoking Rates</span>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="section-card">
        <div class="section-title">✅ Prevention Tips for Health Workers</div>
        <p>Share these with patients during screening camps:</p>
        <p>🥗 <b>Diet:</b> Reduce white rice, refined flour, and sugary drinks. Increase vegetables, lentils, and whole grains.</p>
        <p>🚶 <b>Activity:</b> Even 30 minutes of walking daily reduces diabetes risk by up to 30%.</p>
        <p>⚖️ <b>Weight:</b> Losing 5-7% of body weight significantly reduces risk in prediabetic individuals.</p>
        <p>🩸 <b>Screening:</b> Anyone over 40, or with a family history, should be screened annually.</p>
        <p>💊 <b>Medication:</b> If diagnosed, consistent medication adherence prevents complications.</p>
    </div>
    """, unsafe_allow_html=True)

# Footer
st.markdown("""
<div class="footer">
    <b style="font-size:1.1rem">🩺 GlucoseGuard PK</b><br>
    AI-Assisted Diabetes Triage & Referral Support System for Pakistan<br>
    <span style="font-size:0.8rem;opacity:0.7">
    ⚠️ For screening support only. Does not replace professional clinical diagnosis or medical advice.
    </span>
</div>
""", unsafe_allow_html=True)
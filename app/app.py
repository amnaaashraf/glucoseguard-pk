import streamlit as st
import joblib
import numpy as np
import pandas as pd
import shap
import time

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

.stApp {
    background-color: #f0faf9;
}

/* Top banner */
.top-banner {
    background: linear-gradient(135deg, #0d9488, #0f766e);
    padding: 2.5rem 3rem;
    border-radius: 0 0 30px 30px;
    margin: -1rem -1rem 2rem -1rem;
    display: flex;
    align-items: center;
    justify-content: space-between;
    box-shadow: 0 8px 32px rgba(13, 148, 136, 0.3);
}
.banner-left h1 {
    color: white;
    font-size: 2.4rem;
    font-weight: 800;
    margin: 0;
    letter-spacing: -0.5px;
}
.banner-left p {
    color: rgba(255,255,255,0.85);
    font-size: 1rem;
    margin: 0.3rem 0 0 0;
}
.banner-badge {
    background: rgba(255,255,255,0.15);
    border: 1px solid rgba(255,255,255,0.3);
    color: white;
    padding: 0.5rem 1.2rem;
    border-radius: 50px;
    font-size: 0.85rem;
    font-weight: 600;
}

/* Stat cards */
.stat-row {
    display: flex;
    gap: 1rem;
    margin-bottom: 2rem;
}
.stat-card {
    background: white;
    border-radius: 16px;
    padding: 1.2rem 1.5rem;
    flex: 1;
    box-shadow: 0 2px 12px rgba(0,0,0,0.06);
    border-top: 4px solid #0d9488;
}
.stat-card h3 {
    color: #0d9488;
    font-size: 1.6rem;
    font-weight: 800;
    margin: 0;
}
.stat-card p {
    color: #6b7280;
    font-size: 0.85rem;
    margin: 0.2rem 0 0 0;
}

/* Section card */
.section-card {
    background: white;
    border-radius: 20px;
    padding: 2rem;
    box-shadow: 0 2px 16px rgba(0,0,0,0.06);
    margin-bottom: 1.5rem;
}
.section-title {
    color: #0f766e;
    font-size: 1.1rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 1px;
    margin-bottom: 1.2rem;
    padding-bottom: 0.8rem;
    border-bottom: 2px solid #ccfbf1;
}

/* Risk results */
.result-low {
    background: linear-gradient(135deg, #ecfdf5, #d1fae5);
    border: 2px solid #10b981;
    border-radius: 20px;
    padding: 2rem;
    text-align: center;
}
.result-medium {
    background: linear-gradient(135deg, #fffbeb, #fef3c7);
    border: 2px solid #f59e0b;
    border-radius: 20px;
    padding: 2rem;
    text-align: center;
}
.result-high {
    background: linear-gradient(135deg, #fff1f2, #fee2e2);
    border: 2px solid #ef4444;
    border-radius: 20px;
    padding: 2rem;
    text-align: center;
}
.result-title {
    font-size: 1.8rem;
    font-weight: 800;
    margin: 0.5rem 0;
}
.result-score {
    font-size: 3.5rem;
    font-weight: 900;
    margin: 0;
}
.action-box {
    background: white;
    border-radius: 12px;
    padding: 1rem 1.5rem;
    margin-top: 1rem;
    font-size: 0.95rem;
    font-weight: 500;
}

/* Driver cards */
.driver-card {
    background: #f8fafc;
    border-radius: 12px;
    padding: 0.9rem 1.2rem;
    margin: 0.5rem 0;
    display: flex;
    align-items: center;
    gap: 0.8rem;
    border-left: 4px solid #0d9488;
}

/* Footer */
.footer {
    background: #134e4a;
    color: rgba(255,255,255,0.7);
    text-align: center;
    padding: 1.5rem;
    border-radius: 16px;
    margin-top: 2rem;
    font-size: 0.85rem;
}

/* Streamlit overrides */
.stButton > button {
    background: linear-gradient(135deg, #0d9488, #0f766e) !important;
    color: white !important;
    border: none !important;
    border-radius: 12px !important;
    padding: 0.8rem 2rem !important;
    font-size: 1.05rem !important;
    font-weight: 700 !important;
    width: 100% !important;
    box-shadow: 0 4px 15px rgba(13,148,136,0.4) !important;
    transition: all 0.3s ease !important;
}
.stSlider > div > div > div > div {
    background: #0d9488 !important;
}
div[data-testid="stSidebar"] {
    background: #134e4a !important;
}
div[data-testid="stSidebar"] * {
    color: white !important;
}
</style>
""", unsafe_allow_html=True)

# Load model
model = joblib.load('models/glucoseguard_model.pkl')
explainer = joblib.load('models/shap_explainer.pkl')

# Top Banner
st.markdown("""
<div class="top-banner">
    <div class="banner-left">
        <h1>🩺 GlucoseGuard PK</h1>
        <p>AI-Assisted Diabetes Triage & Referral Support System for Pakistan</p>
    </div>
    <div class="banner-badge">🇵🇰 Built for Community Health Workers</div>
</div>
""", unsafe_allow_html=True)

# Stats row
st.markdown("""
<div class="stat-row">
    <div class="stat-card">
        <h3>33M+</h3>
        <p>Diabetics in Pakistan</p>
    </div>
    <div class="stat-card">
        <h3>50%</h3>
        <p>Cases Undiagnosed</p>
    </div>
    <div class="stat-card">
        <h3>3rd</h3>
        <p>Highest in the World</p>
    </div>
    <div class="stat-card">
        <h3>80K+</h3>
        <p>Records Used to Train AI</p>
    </div>
</div>
""", unsafe_allow_html=True)

# Sidebar
st.sidebar.markdown("## 🏥 GlucoseGuard PK")
st.sidebar.markdown("---")
mode = st.sidebar.radio("Select Mode",
    ["🔍 Single Patient Screening", "👥 Population Summary"])
st.sidebar.markdown("---")
st.sidebar.markdown("""
**Risk Level Guide**

🟢 **Low Risk** — Under 30%  
Lifestyle advice only

🟡 **Medium Risk** — 30–60%  
Retest within 1 month

🔴 **High Risk** — Above 60%  
Refer for HbA1c immediately
""")
st.sidebar.markdown("---")
st.sidebar.markdown("""
⚠️ *This tool supports clinical 
decision-making and does not 
replace professional diagnosis.*
""")

# SINGLE PATIENT MODE
if "Single" in mode:

    st.markdown("""
    <div class="section-card">
        <div class="section-title">📋 Patient Information</div>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("**🔢 Demographics**")
        age = st.select_slider("Age Group",
            options={
                1: "18–24", 2: "25–29", 3: "30–34",
                4: "35–39", 5: "40–44", 6: "45–49",
                7: "50–54", 8: "55–59", 9: "60–64",
                10: "65–69", 11: "70–74", 12: "75–79", 13: "80+"
            }.keys(),
            format_func=lambda x: {
                1: "18–24", 2: "25–29", 3: "30–34",
                4: "35–39", 5: "40–44", 6: "45–49",
                7: "50–54", 8: "55–59", 9: "60–64",
                10: "65–69", 11: "70–74", 12: "75–79", 13: "80+"
            }[x], value=5)
        sex = st.selectbox("Sex", [0, 1],
            format_func=lambda x: "Male" if x==1 else "Female")
        bmi = st.slider("BMI", 10, 80, 25,
            help="Body Mass Index. Normal: 18.5–24.9")

    with col2:
        st.markdown("**🩸 Clinical Indicators**")
        highbp = st.selectbox("High Blood Pressure?", [0, 1],
            format_func=lambda x: "Yes" if x==1 else "No")
        highchol = st.selectbox("High Cholesterol?", [0, 1],
            format_func=lambda x: "Yes" if x==1 else "No")
        cholcheck = st.selectbox("Cholesterol Checked (5 yrs)?", [0, 1],
            format_func=lambda x: "Yes" if x==1 else "No")
        heartdisease = st.selectbox("Heart Disease History?", [0, 1],
            format_func=lambda x: "Yes" if x==1 else "No")
        stroke = st.selectbox("Stroke History?", [0, 1],
            format_func=lambda x: "Yes" if x==1 else "No")

    with col3:
        st.markdown("**🏃 Lifestyle & Wellbeing**")
        smoker = st.selectbox("Smoker?", [0, 1],
            format_func=lambda x: "Yes" if x==1 else "No")
        physactivity = st.selectbox("Physically Active?", [0, 1],
            format_func=lambda x: "Yes" if x==1 else "No")
        fruits = st.selectbox("Eats Fruits Daily?", [0, 1],
            format_func=lambda x: "Yes" if x==1 else "No")
        veggies = st.selectbox("Eats Vegetables Daily?", [0, 1],
            format_func=lambda x: "Yes" if x==1 else "No")
        hvyalcohol = st.selectbox("Heavy Alcohol Use?", [0, 1],
            format_func=lambda x: "Yes" if x==1 else "No")

    st.markdown("---")
    col4, col5 = st.columns(2)
    with col4:
        genhlth = st.select_slider("General Health",
            options=[1,2,3,4,5],
            format_func=lambda x: {
                1:"Excellent",2:"Very Good",
                3:"Good",4:"Fair",5:"Poor"}[x], value=3)
        diffwalk = st.selectbox("Difficulty Walking?", [0, 1],
            format_func=lambda x: "Yes" if x==1 else "No")
    with col5:
        menthlth = st.slider("Poor Mental Health Days (last 30)", 0, 30, 0)
        physhlth = st.slider("Poor Physical Health Days (last 30)", 0, 30, 0)
        anyhealthcare = st.selectbox("Has Healthcare Access?", [0, 1],
            format_func=lambda x: "Yes" if x==1 else "No")
        nodocbccost = st.selectbox("Avoided Doctor due to Cost?", [0, 1],
            format_func=lambda x: "Yes" if x==1 else "No")

    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    if st.button("🔍 Assess Diabetes Risk"):
        with st.spinner("🧠 Analyzing patient profile..."):
            time.sleep(1.5)

        input_data = pd.DataFrame([[
            highbp, highchol, cholcheck, bmi, smoker,
            stroke, heartdisease, physactivity, fruits,
            veggies, hvyalcohol, anyhealthcare, nodocbccost,
            genhlth, menthlth, physhlth, diffwalk, sex,
            age, 4, 5
        ]], columns=[
            'HighBP','HighChol','CholCheck','BMI','Smoker',
            'Stroke','HeartDiseaseorAttack','PhysActivity','Fruits',
            'Veggies','HvyAlcoholConsump','AnyHealthcare','NoDocbcCost',
            'GenHlth','MentHlth','PhysHlth','DiffWalk','Sex',
            'Age','Education','Income'
        ])

        prob = model.predict_proba(input_data)[0][1]
        risk_pct = round(prob * 100, 1)

        if prob < 0.3:
            css_class = "result-low"
            risk_label = "🟢 LOW RISK"
            action = "✅ Provide lifestyle counselling. Schedule rescreen in 12 months."
            emoji = "😊"
        elif prob < 0.6:
            css_class = "result-medium"
            risk_label = "🟡 MEDIUM RISK"
            action = "⚠️ Repeat fasting glucose test within 1 month. Recommend dietary changes."
            emoji = "⚠️"
        else:
            css_class = "result-high"
            risk_label = "🔴 HIGH RISK"
            action = "🚨 Refer for HbA1c testing within 2 weeks. Immediate clinical attention required."
            emoji = "🚨"

        st.markdown("---")
        st.markdown("## 📊 Assessment Results")

        col_a, col_b = st.columns(2)

        with col_a:
            progress = st.progress(0)
            for i in range(int(risk_pct)):
                time.sleep(0.015)
                progress.progress(i+1)

            st.markdown(f"""
            <div class="{css_class}">
                <div class="result-score">{risk_pct}%</div>
                <div class="result-title">{risk_label}</div>
                <div class="action-box">{action}</div>
            </div>
            """, unsafe_allow_html=True)

        with col_b:
            st.markdown("### 🔍 Key Risk Drivers")
            st.markdown("*Factors most influencing this patient's risk:*")

            shap_vals = explainer.shap_values(input_data)
            feature_df = pd.DataFrame({
                'Feature': input_data.columns,
                'Impact': shap_vals[0],
                'Abs': abs(shap_vals[0])
            }).nlargest(5, 'Abs')

            labels = {
                'HighBP': 'High Blood Pressure',
                'HighChol': 'High Cholesterol',
                'BMI': 'Body Mass Index',
                'Smoker': 'Smoking',
                'HeartDiseaseorAttack': 'Heart Disease',
                'PhysActivity': 'Physical Activity',
                'GenHlth': 'General Health',
                'Age': 'Age Group',
                'Stroke': 'Stroke History',
                'Fruits': 'Fruit Intake',
                'Veggies': 'Vegetable Intake',
                'MentHlth': 'Mental Health',
                'PhysHlth': 'Physical Health Days',
                'DiffWalk': 'Difficulty Walking',
                'CholCheck': 'Cholesterol Check',
                'HvyAlcoholConsump': 'Alcohol Use',
                'AnyHealthcare': 'Healthcare Access',
                'NoDocbcCost': 'Avoided Doctor (Cost)',
                'Sex': 'Sex',
            }

            for _, row in feature_df.iterrows():
                direction = "🔺 Increases risk" if row['Impact'] > 0 else "🔻 Decreases risk"
                color = "#ef4444" if row['Impact'] > 0 else "#10b981"
                name = labels.get(row['Feature'], row['Feature'])
                st.markdown(f"""
                <div class="driver-card" style="border-left-color: {color}">
                    <div>
                        <b style="color: #1f2937">{name}</b><br>
                        <span style="color: {color}; font-size: 0.85rem">{direction}</span>
                    </div>
                </div>
                """, unsafe_allow_html=True)

# POPULATION MODE
elif "Population" in mode:
    st.markdown("""
    <div class="section-card">
        <div class="section-title">👥 Population Screening Summary</div>
        <p style="color: #6b7280">Upload a CSV file of screened patients to get 
        population-level insights and triage prioritization.</p>
    </div>
    """, unsafe_allow_html=True)

    uploaded_file = st.file_uploader("Upload Patient CSV", type=['csv'])

    if uploaded_file:
        with st.spinner("Analyzing population data..."):
            pop_df = pd.read_csv(uploaded_file)
            probs = model.predict_proba(pop_df)[:,1]
            time.sleep(1)

        low = sum(probs < 0.3)
        medium = sum((probs >= 0.3) & (probs < 0.6))
        high = sum(probs >= 0.6)
        total = len(probs)

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("👥 Total Screened", total)
        col2.metric("🟢 Low Risk", f"{low} ({round(low/total*100)}%)")
        col3.metric("🟡 Medium Risk", f"{medium} ({round(medium/total*100)}%)")
        col4.metric("🔴 High Risk", f"{high} ({round(high/total*100)}%)")

        st.markdown("### 📈 Population Risk Factors")
        risk_cols = ['HighBP','HighChol','BMI','Smoker','PhysActivity']
        available = [c for c in risk_cols if c in pop_df.columns]
        if available:
            st.bar_chart(pop_df[available].mean())

# Footer
st.markdown("""
<div class="footer">
    🩺 <b>GlucoseGuard PK</b> — AI-Assisted Diabetes Triage for Pakistan's Community Health Workers<br>
    <span style="font-size:0.8rem">⚠️ For screening support only. Does not replace clinical diagnosis or professional medical advice.</span>
</div>
""", unsafe_allow_html=True)
import streamlit as st
import pandas as pd
import pickle
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer

# --- LOAD MODEL ---
@st.cache_resource
def load_models():
    try:
        with open('lr_model.pkl', 'rb') as f:
            lr = pickle.load(f)
        with open('rf_model.pkl', 'rb') as f:
            rf = pickle.load(f)
        return lr, rf
    except:
        st.error("Model tidak ditemukan. Upload `lr_model.pkl` dan `rf_model.pkl`.")
        st.stop()

lr, rf = load_models()

# --- PREPROCESSOR ---
def create_preprocessor():
    cat_cols = ['Parental_Involvement', 'Motivation_Level', 'Peer_Influence', 'Internet_Access', 'Extracurricular_Activities']
    num_cols = ['Hours_Studied', 'Attendance', 'Sleep_Hours', 'Physical_Activity', 'Previous_Scores', 'Tutoring_Sessions']
    return ColumnTransformer([
        ('cat', OneHotEncoder(drop='first', handle_unknown='ignore'), cat_cols),
        ('num', StandardScaler(), num_cols)
    ], sparse_threshold=0)

# --- DUMMY DATA + URUTAN KOLOM ---
dummy_df = pd.DataFrame([
    {'Hours_Studied': 1, 'Attendance': 50, 'Sleep_Hours': 4, 'Physical_Activity': 0, 'Previous_Scores': 50, 'Tutoring_Sessions': 0,
     'Parental_Involvement': 'Low', 'Motivation_Level': 'Low', 'Peer_Influence': 'Negative', 'Internet_Access': 'No', 'Extracurricular_Activities': 'No'},
    {'Hours_Studied': 30, 'Attendance': 100, 'Sleep_Hours': 9, 'Physical_Activity': 6, 'Previous_Scores': 95, 'Tutoring_Sessions': 4,
     'Parental_Involvement': 'High', 'Motivation_Level': 'High', 'Peer_Influence': 'Positive', 'Internet_Access': 'Yes', 'Extracurricular_Activities': 'Yes'}
])

# PAKSA URUTAN KOLOM SAMA
dummy_df = dummy_df[[
    'Hours_Studied', 'Attendance', 'Sleep_Hours', 'Physical_Activity',
    'Previous_Scores', 'Tutoring_Sessions',
    'Parental_Involvement', 'Motivation_Level', 'Peer_Influence',
    'Internet_Access', 'Extracurricular_Activities'
]]

preprocessor = create_preprocessor()
preprocessor.fit(dummy_df)

# --- UI ---
st.set_page_config(page_title="Prediksi Performa Nilai Siswa", page_icon="Graduation Cap", layout="wide")
st.title("Prediksi Performa Nilai Siswa")
st.markdown("Menggunakan 11 indikator utama berdasarkan **Akademik & Gaya Hidup** serta **Lingkungan & Psikologis**")
st.markdown("""
**Peringatan**:  
Model menggunakan data dummy dan **SMOTE** (terdapat data sintesis) sehingga mungkin beberapa hasil prediksi kurang realistis.
""")

# --- FORM INPUT ---
with st.form("input_form"):
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Akademik & Gaya Hidup**")
        hours_studied = st.slider("Hours Studied", 1, 50, 23)
        attendance = st.slider("Attendance (%)", 50, 100, 84)
        sleep_hours = st.slider("Sleep Hours", 4, 10, 7)
        physical_activity = st.slider("Physical Activity (0-6)", 0, 6, 3)
        previous_scores = st.slider("Previous Scores", 50, 100, 73)
        tutoring_sessions = st.slider("Tutoring Sessions", 0, 5, 1)
    with col2:
        st.markdown("**Lingkungan & Psikologis**")
        parental_involvement = st.selectbox("Parental Involvement", ['Low', 'Medium', 'High'])
        motivation_level = st.selectbox("Motivation Level", ['Low', 'Medium', 'High'])
        peer_influence = st.selectbox("Peer Influence", ['Negative', 'Neutral', 'Positive'])
        internet_access = st.selectbox("Internet Access", ['No', 'Yes'])
        extracurricular_activities = st.selectbox("Extracurricular Activities", ['No', 'Yes'])
    
    model_choice = st.selectbox("Model", ["Random Forest", "Logistic Regression"])
    submitted = st.form_submit_button("Prediksi")

# --- PREDIKSI (DI LUAR FORM!) ---
if submitted:
    # --- HITUNG SKOR MANUAL ---
    score = 0
    if hours_studied >= 25: score += 2
    if attendance >= 80: score += 2
    if sleep_hours >= 7: score += 1
    if physical_activity >= 3: score += 1
    if previous_scores >= 70: score += 2
    if tutoring_sessions >= 2: score += 1
    if parental_involvement == ['Medium','High']: score += 1
    if motivation_level == ['Medium','High']: score += 2
    if peer_influence == 'Positive': score += 1
    if internet_access == 'Yes': score += 1
    if extracurricular_activities == 'Yes': score += 1

    # --- INPUT DATAFRAME (URUTAN SAMA DENGAN DUMMY) ---
    input_data = {
        'Hours_Studied': hours_studied,
        'Attendance': attendance,
        'Sleep_Hours': sleep_hours,
        'Physical_Activity': physical_activity,
        'Previous_Scores': previous_scores,
        'Tutoring_Sessions': tutoring_sessions,
        'Parental_Involvement': parental_involvement,
        'Motivation_Level': motivation_level,
        'Peer_Influence': peer_influence,
        'Internet_Access': internet_access,
        'Extracurricular_Activities': extracurricular_activities
    }
    input_df = pd.DataFrame([input_data])
    input_df = input_df[[
        'Hours_Studied', 'Attendance', 'Sleep_Hours', 'Physical_Activity',
        'Previous_Scores', 'Tutoring_Sessions',
        'Parental_Involvement', 'Motivation_Level', 'Peer_Influence',
        'Internet_Access', 'Extracurricular_Activities'
    ]]

    # --- TRANSFORM & PREDIKSI ---
    X_input = preprocessor.transform(input_df)
    model = rf if "Random Forest" in model_choice else lr
    pred = model.predict(X_input)[0]

    # --- HASIL AKHIR ---
    st.subheader("Hasil Prediksi")
    if score >= 10:
        st.success("LULUS (nilai di atas passing grade)")
        st.balloons()
    elif score <= 4:
        st.error("RISIKO TIDAK LULUS (nilai di bawah passing grade)")
    else:
        if pred == 'Above Threshold':
            st.success("LULUS (nilai di atas passing grade)")
        else:
            st.error("RISIKO TIDAK LULUS (nilai di bawah passing grade)")
        st.caption(f"Model: {'LULUS' if pred == 'Above Threshold' else 'TIDAK LULUS'} | Skor: {score}/15")

    # --- REKOMENDASI ---
    st.subheader("Rekomendasi")
    if hours_studied < 25: st.warning("Tingkatkan jam belajar ≥25 jam")
    if attendance < 80: st.warning("Tingkatkan kehadiran ≥80%")
    if sleep_hours < 7: st.warning("Tidur ≥8 jam")
    if physical_activity < 3: st.warning("Olahraga ≥3x")
    if tutoring_sessions <= 2 and previous_scores < 85: st.warning("Nilai sebelumnya < 85, ikuti program tutoring")
    if motivation_level == 'Low': st.warning("Tingkatkan motivasi")
    if parental_involvement == 'Low': st.warning("Lebih libatkan orangtua")

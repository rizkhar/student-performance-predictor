import streamlit as st
import pandas as pd
import pickle
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer

# --- LOAD 2 MODEL ---
@st.cache_resource
def load_models():
    with open('lr_model.pkl', 'rb') as f:
        lr = pickle.load(f)
    with open('rf_model.pkl', 'rb') as f:
        rf = pickle.load(f)
    return lr, rf

lr, rf = load_models()

# --- PREPROCESSOR (SESUAI .ipynb) ---
def create_preprocessor():
    cat_cols = [
        'Parental_Involvement', 'Motivation_Level', 'Peer_Influence',
        'Internet_Access', 'Extracurricular_Activities'
    ]
    num_cols = [
        'Hours_Studied', 'Attendance', 'Sleep_Hours',
        'Physical_Activity', 'Previous_Scores', 'Tutoring_Sessions'
    ]
    return ColumnTransformer([
        ('cat', OneHotEncoder(drop='first', handle_unknown='ignore'), cat_cols),
        ('num', StandardScaler(), num_cols)
    ], sparse_threshold=0)

# --- DUMMY DATA UNTUK FIT (FIX ERROR) ---
dummy_df = pd.DataFrame([
    {
        'Hours_Studied': 1, 'Attendance': 50, 'Sleep_Hours': 4,
        'Physical_Activity': 0, 'Previous_Scores': 50, 'Tutoring_Sessions': 0,
        'Parental_Involvement': 'Low', 'Motivation_Level': 'Low',
        'Peer_Influence': 'Negative', 'Internet_Access': 'No',
        'Extracurricular_Activities': 'No'
    },
    {
        'Hours_Studied': 30, 'Attendance': 100, 'Sleep_Hours': 9,
        'Physical_Activity': 6, 'Previous_Scores': 95, 'Tutoring_Sessions': 4,
        'Parental_Involvement': 'High', 'Motivation_Level': 'High',
        'Peer_Influence': 'Positive', 'Internet_Access': 'Yes',
        'Extracurricular_Activities': 'Yes'
    }
])

preprocessor = create_preprocessor()
preprocessor.fit(dummy_df)

# --- UI ---
st.set_page_config(page_title="Prediksi Kelulusan", page_icon="Graduation Cap", layout="wide")
st.title("Prediksi Kelulusan Siswa")
st.markdown("**11 Fitur Terpilih | Hanya 2 Hasil: LULUS / TIDAK LULUS**")

st.markdown("""
**Catatan**:  
Hanya **24.8% siswa** yang lulus (Exam_Score ≥ 70).  
Model menggunakan SMOTE — prediksi bisa **over-optimis**.  
Gunakan rekomendasi untuk **intervensi dini**.
""")

# --- INPUT FORM ---
with st.form("input_form"):
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Faktor Akademik & Gaya Hidup**")
        hours_studied = st.slider("Hours Studied", 1, 50, 23)
        attendance = st.slider("Attendance (%)", 50, 100, 84)
        sleep_hours = st.slider("Sleep Hours", 4, 10, 7)
        physical_activity = st.slider("Physical Activity (0-6)", 0, 6, 3)
        previous_scores = st.slider("Previous Scores", 50, 100, 73)
        tutoring_sessions = st.slider("Tutoring Sessions", 0, 5, 1)
    with col2:
        st.markdown("**Faktor Lingkungan & Psikologis**")
        parental_involvement = st.selectbox("Parental Involvement", ['Low', 'Medium', 'High'])
        motivation_level = st.selectbox("Motivation Level", ['Low', 'Medium', 'High'])
        peer_influence = st.selectbox("Peer Influence", ['Negative', 'Neutral', 'Positive'])
        internet_access = st.selectbox("Internet Access", ['No', 'Yes'])
        extracurricular_activities = st.selectbox("Extracurricular Activities", ['No', 'Yes'])
    
    model_choice = st.selectbox("Pilih Model", ["Random Forest", "Logistic Regression"])
    submitted = st.form_submit_button("Prediksi")

# --- PREDIKSI ---
if submitted:
    # Input DataFrame
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

    # Transform
    X_input = preprocessor.transform(input_df)
    model = rf if "Random Forest" in model_choice else lr
    pred = model.predict(X_input)[0]

    # --- LOGIKA 2 OUTPUT ---
    all_high = (
        hours_studied >= 25 and attendance >= 95 and sleep_hours >= 8 and
        physical_activity >= 5 and previous_scores >= 85 and tutoring_sessions >= 3 and
        parental_involvement == 'High' and motivation_level == 'High' and
        peer_influence == 'Positive' and internet_access == 'Yes' and
        extracurricular_activities == 'Yes'
    )

    all_low = (
        hours_studied <= 10 and attendance <= 70 and sleep_hours <= 5 and
        physical_activity <= 1 and previous_scores <= 60 and tutoring_sessions <= 0 and
        parental_involvement == 'Low' and motivation_level == 'Low' and
        peer_influence == 'Negative'
    )

    # --- OUTPUT ---
    st.subheader("Hasil Prediksi")
    if all_high:
        st.success("LULUS (nilai di atas passing grade)")
        st.balloons()
        st.info("Semua faktor optimal")
    elif all_low:
        st.error("RISIKO TIDAK LULUS (nilai di bawah passing grade)")
        st.info("Terlalu banyak risiko")
    elif pred == 'Above Threshold':
        st.success("LULUS (nilai di atas passing grade)")
    else:
        st.error("RISIKO TIDAK LULUS (nilai di bawah passing grade)")

    # --- REKOMENDASI ---
    st.subheader("Rekomendasi Intervensi")
    if hours_studied < 20: st.warning("Belajar ≥25 jam/minggu")
    if attendance < 80: st.warning("Kehadiran ≥80%")
    if sleep_hours < 8: st.warning("Tidur ≥8 jam/hari")
    if physical_activity < 3: st.warning("Olahraga ≥3x/minggu")
    if tutoring_sessions < 2: st.warning("Tambah bimbingan 2 sesi")
    if motivation_level != 'High': st.warning("Tingkatkan motivasi")
    if parental_involvement != 'High': st.warning("Libatkan orang tua lebih aktif")

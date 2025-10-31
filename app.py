import streamlit as st
import pandas as pd
import pickle
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer

# --- LOAD MODEL ---
@st.cache_resource
def load_models():
    with open('lr_model.pkl', 'rb') as f:
        lr = pickle.load(f)
    with open('rf_model.pkl', 'rb') as f:
        rf = pickle.load(f)
    return lr, rf

lr, rf = load_models()

# --- PREPROCESSOR ---
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

# Dummy data untuk fit (agar semua kategori terdeteksi)
dummy_data = []
# semua nilai rendah
low = {
    'Hours_Studied': 1, 'Attendance': 50, 'Sleep_Hours': 4,
    'Physical_Activity': 0, 'Previous_Scores': 50, 'Tutoring_Sessions': 0,
    'Parental_Involvement': 'Low', 'Motivation_Level': 'Low',
    'Peer_Influence': 'Negative', 'Internet_Access': 'No',
    'Extracurricular_Activities': 'No'
}
# semua nilai tinggi
high = {
    'Hours_Studied': 30, 'Attendance': 100, 'Sleep_Hours': 9,
    'Physical_Activity': 6, 'Previous_Scores': 95, 'Tutoring_Sessions': 4,
    'Parental_Involvement': 'High', 'Motivation_Level': 'High',
    'Peer_Influence': 'Positive', 'Internet_Access': 'Yes',
    'Extracurricular_Activities': 'Yes'
}
dummy_data = [low, high]
dummy_df = pd.DataFrame(dummy_data)

preprocessor = create_preprocessor()
preprocessor.fit(dummy_df)

# --- UI ---
st.set_page_config(page_title="Student Success Predictor", page_icon="Graduation Cap", layout="wide")
st.title("Student Success Predictor")
st.markdown("**11 Fitur Terpilih | Hanya 24.8% Siswa Lulus**")

st.markdown("""
**Peringatan**:  
Data yang digunakan merupakan data dummy, dan indikator kelulusan mungkin kurang realistis karena model menggunakan SMOTE untuk menambahkan data sintetis.
""")

# --- INPUT FORM (SESUAI .ipynb) ---
with st.form("input_form"):
    col1, col2 = st.columns(2)
    with col1:
        hours_studied = st.slider("**Hours Studied**", 1, 50, 23)
        attendance = st.slider("**Attendance (%)**", 50, 100, 84)
        sleep_hours = st.slider("**Sleep Hours**", 4, 10, 7)
        physical_activity = st.slider("**Physical Activity (0-6)**", 0, 6, 3)
        previous_scores = st.slider("**Previous Scores**", 50, 100, 73)
        tutoring_sessions = st.slider("**Tutoring Sessions**", 0, 5, 1)
    with col2:
        parental_involvement = st.selectbox("**Parental Involvement**", ['Low', 'Medium', 'High'])
        motivation_level = st.selectbox("**Motivation Level**", ['Low', 'Medium', 'High'])
        peer_influence = st.selectbox("**Peer Influence**", ['Negative', 'Neutral', 'Positive'])
        internet_access = st.selectbox("**Internet Access**", ['No', 'Yes'])
        extracurricular_activities = st.selectbox("**Extracurricular Activities**", ['No', 'Yes'])
    
    model_choice = st.selectbox("**Pilih Model**", ["Random Forest", "Logistic Regression"])
    submitted = st.form_submit_button("Prediksi")

if submitted:
    # --- INPUT DATAFRAME (URUTAN SAMA DENGAN .ipynb) ---
    input_df = pd.DataFrame([{
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
    }])

    # --- TRANSFORM ---
    X_input = preprocessor.transform(input_df)
    model = rf if "Random Forest" in model_choice else lr
    pred = model.predict(X_input)[0]
    prob_lulus = model.predict_proba(X_input)[0][1]

    # --- REALITY CHECK (SESUAI DATA 24.8%) ---
    critical_risk = (
        hours_studied < 12 or
        attendance < 50 or
        sleep_hours < 6 or
        physical_activity < 2 or
        previous_scores < 65 or
        motivation_level == 'Low'
    )

    # --- OUTPUT ---
    st.subheader("Hasil Prediksi")
    if critical_risk:
        st.error("**RISIKO GAGAL (nilai di bawah passing grade**")
        st.info("Probabilitas lulus mungkin tinggi, **realita hanya 24.8% lulus**.")
    elif pred == 'Above Threshold':
        st.success(f"**LULUS (nilai di atas passing grade)** | Probabilitas lulus: {prob_lulus:.1%}")
    else:
        st.error(f"**RISIKO GAGAL (nilai di bawah passing grade)** | Probabilitas lulus: {prob_lulus:.1%}")

    # --- REKOMENDASI ---
    st.subheader("Rekomendasi Intervensi")
    if hours_studied < 15: st.error("Belajar <15 jam → risiko gagal >90%")
    if attendance < 80: st.error("Kehadiran <80% → hampir pasti gagal")
    if sleep_hours < 7: st.warning("Tidur <7 jam → otak tidak optimal")
    if physical_activity < 3: st.warning("Olahraga <3x/minggu → stamina rendah")
    if motivation_level == 'Low': st.error("Motivasi rendah → butuh konseling segera")
    if tutoring_sessions < 2: st.warning("Tambah bimbingan 2 sesi/minggu")

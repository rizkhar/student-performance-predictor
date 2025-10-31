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

# --- BUAT PREPROCESSOR + DUMMY DATA UNTUK FIT ---
def create_preprocessor_with_fit():
    cat_cols = [
        'Parental_Involvement', 'Motivation_Level', 'Peer_Influence',
        'Internet_Access', 'Extracurricular_Activities'
    ]
    num_cols = [
        'Hours_Studied', 'Attendance', 'Sleep_Hours',
        'Physical_Activity', 'Previous_Scores', 'Tutoring_Sessions'
    ]
    
    preprocessor = ColumnTransformer([
        ('cat', OneHotEncoder(drop='first', handle_unknown='ignore'), cat_cols),
        ('num', StandardScaler(), num_cols)
    ], sparse_threshold=0)

    # Dummy data untuk fit
    dummy_data = pd.DataFrame([
        {
            'Hours_Studied': 20, 'Attendance': 80, 'Sleep_Hours': 7,
            'Physical_Activity': 3, 'Previous_Scores': 70, 'Tutoring_Sessions': 1,
            'Parental_Involvement': 'Low', 'Motivation_Level': 'Low',
            'Peer_Influence': 'Negative', 'Internet_Access': 'No',
            'Extracurricular_Activities': 'No'
        },
        {
            'Hours_Studied': 25, 'Attendance': 95, 'Sleep_Hours': 8,
            'Physical_Activity': 5, 'Previous_Scores': 85, 'Tutoring_Sessions': 3,
            'Parental_Involvement': 'High', 'Motivation_Level': 'High',
            'Peer_Influence': 'Positive', 'Internet_Access': 'Yes',
            'Extracurricular_Activities': 'Yes'
        }
    ])

    # Fit di dummy agar encoder tahu semua kategori
    preprocessor.fit(dummy_data)
    return preprocessor

# Load awal
preprocessor = create_preprocessor_with_fit()

# --- UI ---
st.set_page_config(page_title="Student Predictor", page_icon="Graduation Cap", layout="wide")
st.title("Student Success Predictor")
st.markdown("**11 Fitur Terpilih** | Tanpa `preprocessor.pkl` | **FIXED**")

model_choice = st.selectbox("Model", ["Random Forest", "Logistic Regression"])
model = rf if "Random Forest" in model_choice else lr

# --- INPUT FORM ---
with st.form("input_form"):
    col1, col2 = st.columns(2)
    with col1:
        hours_studied = st.slider("Hours_Studied", 1, 50, 23)
        attendance = st.slider("Attendance (%)", 50, 100, 84)
        sleep_hours = st.slider("Sleep_Hours", 4, 10, 7)
        physical_activity = st.slider("Physical_Activity (0-6)", 0, 6, 3)
        previous_scores = st.slider("Previous_Scores", 50, 100, 73)
        tutoring_sessions = st.slider("Tutoring_Sessions", 0, 5, 1)
    with col2:
        parental_involvement = st.selectbox("Parental_Involvement", ['Low', 'Medium', 'High'])
        motivation_level = st.selectbox("Motivation_Level", ['Low', 'Medium', 'High'])
        peer_influence = st.selectbox("Peer_Influence", ['Negative', 'Neutral', 'Positive'])
        internet_access = st.selectbox("Internet_Access", ['No', 'Yes'])
        extracurricular_activities = st.selectbox("Extracurricular_Activities", ['No', 'Yes'])
    
    submitted = st.form_submit_button("Prediksi")

if submitted:
    # Input DataFrame
    input_df = pd.DataFrame({
        'Hours_Studied': [hours_studied],
        'Attendance': [attendance],
        'Sleep_Hours': [sleep_hours],
        'Physical_Activity': [physical_activity],
        'Previous_Scores': [previous_scores],
        'Tutoring_Sessions': [tutoring_sessions],
        'Parental_Involvement': [parental_involvement],
        'Motivation_Level': [motivation_level],
        'Peer_Influence': [peer_influence],
        'Internet_Access': [internet_access],
        'Extracurricular_Activities': [extracurricular_activities]
    })

    # Transform (sudah difit di dummy)
    X_input = preprocessor.transform(input_df)

    # Prediksi
    pred = model.predict(X_input)[0]
    prob = model.predict_proba(X_input)[0].max()

    # Output
    st.subheader("Hasil")
    if pred == 'Above Threshold':
        st.success(f"LULUS | Peluang: {prob:.1%}")
    else:
        st.error(f"RISIKO GAGAL | Peluang: {prob:.1%}")

    st.subheader("Rekomendasi")
    if sleep_hours < 7: st.warning("Tidur 7 jam")
    if physical_activity < 3: st.warning("Olahraga ≥3x/minggu")
    if motivation_level == 'Low': st.warning("Konseling motivasi")
    if attendance < 85: st.warning("Tingkatkan kehadiran")
    if tutoring_sessions < 2: st.warning("Tambah sesi bimbingan")

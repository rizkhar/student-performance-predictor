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
        'Parental_Involvement', 'Access_to_Resources', 'Extracurricular_Activities',
        'Motivation_Level', 'Internet_Access', 'Family_Income', 'Teacher_Quality',
        'School_Type', 'Peer_Influence', 'Learning_Disabilities',
        'Parental_Education_Level', 'Distance_from_Home', 'Gender'
    ]
    num_cols = [
        'Hours_Studied', 'Attendance', 'Sleep_Hours', 'Previous_Scores',
        'Tutoring_Sessions', 'Physical_Activity'
    ]
    
    preprocessor = ColumnTransformer([
        ('cat', OneHotEncoder(drop='first', handle_unknown='ignore'), cat_cols),
        ('num', StandardScaler(), num_cols)
    ])
    return preprocessor

# --- APP UI ---
st.set_page_config(page_title="Student Predictor", page_icon="Graduation Cap", layout="centered")
st.title("Student Success Predictor")
st.markdown("**Prediksi LULUS / GAGAL** berdasarkan pola hidup. Pilih model:")

model_choice = st.selectbox("Model", ["Random Forest (Lebih Akurat)", "Logistic Regression (Lebih Cepat)"])
model = rf if "Random Forest" in model_choice else lr

# --- INPUT FORM ---
with st.form("input_form"):
    col1, col2 = st.columns(2)
    with col1:
        hours_studied = st.slider("Jam Belajar/Minggu", 1, 50, 23)
        attendance = st.slider("Kehadiran (%)", 50, 100, 84)
        sleep_hours = st.slider("Jam Tidur/Hari", 4, 10, 7)
    with col2:
        physical_activity = st.slider("Olahraga (0-6x/minggu)", 0, 6, 3)
        motivation = st.selectbox("Motivasi", ['Low', 'Medium', 'High'])
        previous_scores = st.slider("Skor Sebelumnya", 50, 100, 73)

    # Default values (sama seperti training)
    defaults = {
        'Parental_Involvement': 'Medium', 'Access_to_Resources': 'Medium',
        'Extracurricular_Activities': 'Yes', 'Internet_Access': 'Yes',
        'Tutoring_Sessions': 1, 'Family_Income': 'Medium',
        'Teacher_Quality': 'Medium', 'School_Type': 'Public',
        'Peer_Influence': 'Neutral', 'Learning_Disabilities': 'No',
        'Parental_Education_Level': 'College', 'Distance_from_Home': 'Near',
        'Gender': 'Male'
    }

    submitted = st.form_submit_button("Prediksi")

if submitted:
    # Buat input DataFrame
    input_dict = {
        'Hours_Studied': hours_studied, 'Attendance': attendance,
        'Sleep_Hours': sleep_hours, 'Previous_Scores': previous_scores,
        'Physical_Activity': physical_activity, 'Motivation_Level': motivation,
        **defaults
    }
    input_df = pd.DataFrame([input_dict])

    # Preprocessing langsung
    preprocessor = create_preprocessor()
    # Fit di input
    X_input = preprocessor.fit_transform(input_df)

    # Prediksi
    pred = model.predict(X_input)[0]
    prob = model.predict_proba(X_input)[0].max()

    # Output
    st.subheader("Hasil Prediksi")
    if pred == 'Above Threshold':
        st.success(f"LULUS | Peluang: {prob:.1%}")
    else:
        st.error(f"RISIKO TIDAK LULUS | Peluang: {prob:.1%}")

    st.subheader("Rekomendasi")
    if sleep_hours < 7:
        st.warning("Tidur **7 jam/hari** → +12 poin")
    if physical_activity < 3:
        st.warning("Olahraga **≥3x/minggu** → 80% lulus")
    if motivation == 'Low':
        st.warning("Konseling motivasi → 3x peluang lulus")
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
st.set_page_config(page_title="Prediksi Kelulusan", page_icon="Graduation Cap", layout="wide")
st.title("Prediksi Kelulusan Siswa")
st.markdown("**11 Fitur | Hanya 2 Hasil: LULUS / TIDAK LULUS**")
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
        physical_activity = st

"""
🎓 Student Exam Score Prediction — Hugging Face Spaces
Model: testscore_model.keras (Deep Learning)
Bağımlılık: sadece models/testscore_model.keras
"""

from __future__ import annotations
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import tensorflow as tf

st.set_page_config(
    page_title="🎓 Student Exam Score Prediction",
    page_icon="🎓",
    layout="wide",
)

# ─── Model Yolu ───────────────────────────────────────────────────────────────
MODEL_PATH = Path("models/testscore_model.keras")

# ─── Sabitler ─────────────────────────────────────────────────────────────────
CATEGORY_OPTIONS = {
    "gender":          ["female", "male", "other"],
    "course":          ["b.com", "b.sc", "b.tech", "ba", "bba", "bca", "diploma"],
    "internet_access": ["yes", "no"],
    "study_method":    ["coaching", "group study", "mixed", "online videos", "self-study"],
    "sleep_quality":   ["poor", "average", "good"],
    "facility_rating": ["low", "medium", "high"],
    "exam_difficulty": ["easy", "moderate", "hard"],
}

# ─── Feature sırası (notebook pd.get_dummies drop_first=True ile aynı) ───────
FEATURE_COLS = [
    "study_hours",
    "class_attendance",
    "sleep_hours",
    "gender_male",
    "gender_other",
    "course_b.sc",
    "course_b.tech",
    "course_ba",
    "course_bba",
    "course_bca",
    "course_diploma",
    "sleep_quality_good",
    "sleep_quality_poor",
    "study_method_group study",
    "study_method_mixed",
    "study_method_online videos",
    "study_method_self-study",
    "facility_rating_low",
    "facility_rating_medium",
    "exam_difficulty_hard",
    "exam_difficulty_moderate",
]

# ─── StandardScaler parametreleri (notebook'tan alındı) ──────────────────────
SCALER_MEAN = np.array([
    4.002337406507937,
    71.98726138412698,
    7.072758031746033,
    0.33427460317460317,
    0.3350746031746032,
    0.17706984126984127,
    0.2083111111111111,
    0.09839523809523809,
    0.12006984126984127,
    0.14082698412698413,
    0.07924444444444445,
    0.33823650793650795,
    0.33916666666666667,
    0.19525238095238096,
    0.19537460317460317,
    0.1921857142857143,
    0.20814444444444444,
    0.3371079365079365,
    0.3398126984126984,
    0.1579015873015873,
    0.5618761904761905,
], dtype=np.float32)

SCALER_SCALE = np.array([
    2.35987841557206,
    17.430083991449198,
    1.74480960682091,
    0.47173625347970105,
    0.47201653941571303,
    0.38172779906959153,
    0.40610047044883535,
    0.29784830906926346,
    0.32504318864925685,
    0.34784298852885087,
    0.2701199038744383,
    0.47310947214722365,
    0.473426487312327,
    0.3963948646044841,
    0.39648879884426685,
    0.39401822992116,
    0.40598070728959273,
    0.4727220913510363,
    0.4736454670005603,
    0.36464870221793805,
    0.4961565650600163,
], dtype=np.float32)

# ─── Model Yükleme ────────────────────────────────────────────────────────────
def load_model():
    if not MODEL_PATH.exists():
        return None
    return tf.keras.models.load_model(str(MODEL_PATH))

# ─── Input Hazırlama ──────────────────────────────────────────────────────────
def prepare_input(payload: dict) -> np.ndarray:
    row = {col: 0.0 for col in FEATURE_COLS}

    row["study_hours"]      = float(payload["study_hours"])
    row["class_attendance"] = float(payload["class_attendance"])
    row["sleep_hours"]      = float(payload["sleep_hours"])

    gender = payload["gender"]
    if gender == "male":
        row["gender_male"] = 1.0
    elif gender == "other":
        row["gender_other"] = 1.0

    course = payload["course"]
    course_key = f"course_{course}"
    if course_key in row:
        row[course_key] = 1.0

    sq = payload["sleep_quality"]
    if sq == "good":
        row["sleep_quality_good"] = 1.0
    elif sq == "poor":
        row["sleep_quality_poor"] = 1.0

    sm = payload["study_method"]
    sm_key = f"study_method_{sm}"
    if sm_key in row:
        row[sm_key] = 1.0

    fr = payload["facility_rating"]
    if fr == "low":
        row["facility_rating_low"] = 1.0
    elif fr == "medium":
        row["facility_rating_medium"] = 1.0

    ed = payload["exam_difficulty"]
    if ed == "hard":
        row["exam_difficulty_hard"] = 1.0
    elif ed == "moderate":
        row["exam_difficulty_moderate"] = 1.0

    x = np.array([row[col] for col in FEATURE_COLS], dtype=np.float32)
    x = (x - SCALER_MEAN) / SCALER_SCALE
    return x.reshape(1, -1)

# ─── Yardımcı ─────────────────────────────────────────────────────────────────
def classify_performance(score: float) -> str:
    if score >= 90: return "Excellent 🏆"
    if score >= 80: return "Very Good 🌟"
    if score >= 70: return "Good 👍"
    if score >= 60: return "Average 📚"
    return "Poor ⚠️"

def assign_grade(score: float) -> str:
    if score >= 90: return "A"
    if score >= 80: return "B"
    if score >= 70: return "C"
    if score >= 60: return "D"
    return "F"

def create_gauge(score: float) -> go.Figure:
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=score,
        number={"suffix": " / 100", "font": {"size": 36}},
        gauge={
            "axis": {"range": [0, 100], "tickwidth": 1},
            "bar":  {"color": "#1890ff", "thickness": 0.3},
            "steps": [
                {"range": [0,  60], "color": "#fff1f0"},
                {"range": [60, 70], "color": "#fff7e6"},
                {"range": [70, 80], "color": "#feffe6"},
                {"range": [80, 90], "color": "#f6ffed"},
                {"range": [90,100], "color": "#d9f7be"},
            ],
            "threshold": {
                "line": {"color": "#1890ff", "width": 4},
                "thickness": 0.75, "value": score,
            },
        },
        title={"text": "Predicted Exam Score", "font": {"size": 18}},
    ))
    fig.update_layout(height=280, margin=dict(t=60, b=0, l=30, r=30))
    return fig

# ─── Sidebar ──────────────────────────────────────────────────────────────────
def build_sidebar() -> dict:
    with st.sidebar:
        st.title("🎓 Student Score Predictor")
        st.markdown("---")
        st.markdown("### 👤 Öğrenci Bilgileri")
        gender           = st.selectbox("Gender",            CATEGORY_OPTIONS["gender"])
        course           = st.selectbox("Course",            CATEGORY_OPTIONS["course"])
        study_hours      = st.slider("Study Hours / Day",    0.0,  7.91,  4.0, 0.01)
        class_attendance = st.slider("Class Attendance (%)", 40.6, 99.4,  72.0, 0.1)
        sleep_hours      = st.slider("Sleep Hours / Day",    4.1,   9.9,  7.0, 0.1)
        st.markdown("### 📋 Diğer Bilgiler")
        sleep_quality   = st.selectbox("Sleep Quality",     CATEGORY_OPTIONS["sleep_quality"])
        study_method    = st.selectbox("Study Method",      CATEGORY_OPTIONS["study_method"])
        facility_rating = st.selectbox("Facility Rating",   CATEGORY_OPTIONS["facility_rating"])
        exam_difficulty = st.selectbox("Exam Difficulty",   CATEGORY_OPTIONS["exam_difficulty"])
        st.markdown("---")
        predict_clicked = st.button(
            "🔍 Predict Score", use_container_width=True, type="primary"
        )

    return {
        "payload": {
            "gender": gender, "course": course,
            "study_hours": study_hours, "class_attendance": class_attendance,
            "sleep_hours": sleep_hours, "sleep_quality": sleep_quality,
            "study_method": study_method,
            "facility_rating": facility_rating, "exam_difficulty": exam_difficulty,
        },
        "predict_clicked": predict_clicked,
    }

# ─── Ana Fonksiyon ────────────────────────────────────────────────────────────
def main():
    model = load_model()
    sidebar = build_sidebar()
    payload = sidebar["payload"]
    predict_clicked = sidebar["predict_clicked"]

    if model is None:
        st.error(
            "⚠️ **Model dosyası bulunamadı!**\n\n"
            "`models/testscore_model.keras` dosyasının mevcut olduğundan emin ol."
        )
        st.stop()

    features = prepare_input(payload)
    score    = float(np.clip(model.predict(features, verbose=0)[0][0], 0, 100))
    perf     = classify_performance(score)
    grade    = assign_grade(score)

    st.title("🎓 Student Exam Score Prediction")
    st.markdown("Soldaki panelden öğrenci bilgilerini gir, anlık tahmin sonuçlarını gör.")
    st.markdown("---")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("📊 Predicted Score", f"{score:.1f} / 100")
    c2.metric("🏅 Performance",     perf)
    c3.metric("📝 Grade",           grade)
    c4.metric("📚 Study Hours",     f"{payload['study_hours']} h/day")

    if predict_clicked:
        st.success("✅ Tahmin başarıyla oluşturuldu!")

    st.markdown("---")

    col_left, col_right = st.columns([1, 1])

    with col_left:
        st.plotly_chart(create_gauge(score), use_container_width=True)

    with col_right:
        st.markdown("#### 📋 Girilen Bilgiler")
        st.dataframe(
            pd.DataFrame({
                "Özellik": [
                    "Cinsiyet", "Bölüm", "Günlük Ders Saati",
                    "Devam Oranı", "Uyku Saati", "Uyku Kalitesi",
                    "Çalışma Yöntemi", "Tesis Kalitesi", "Sınav Zorluğu",
                ],
                "Değer": [
                    payload["gender"],           payload["course"],
                    f"{payload['study_hours']} saat",
                    f"%{payload['class_attendance']}",
                    f"{payload['sleep_hours']} saat",
                    payload["sleep_quality"],
                    payload["study_method"],     payload["facility_rating"],
                    payload["exam_difficulty"],
                ],
            }),
            use_container_width=True,
            hide_index=True,
        )

    st.markdown("---")
    st.markdown("#### ℹ️ Model Bilgisi")
    col_a, col_b, col_c = st.columns(3)
    col_a.metric("Model Tipi",   "Deep Learning (Keras)")
    col_b.metric("Katman Sayısı", "5 Dense")
    col_c.metric("Loss Fonksiyonu", "MSE")

    st.markdown("---")
    st.caption(
        "Powered by TensorFlow/Keras & Streamlit | "
        "[GitHub](https://github.com/tugcesi/predicting-student-test-scores)"
    )

if __name__ == "__main__":
    main()
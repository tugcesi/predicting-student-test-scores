"""
🎓 Student Exam Score Prediction — Hugging Face Spaces
Bağımlılıklar: models/xgb_model.joblib + models/encoders.joblib
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(
    page_title="🎓 Student Exam Score Prediction",
    page_icon="🎓",
    layout="wide",
)

MODEL_PATH    = Path("models/xgb_model.joblib")
ENCODERS_PATH = Path("models/encoders.joblib")

SLEEP_QUALITY_MAP = {"poor": 1, "average": 2, "good": 3}
FACILITY_MAP      = {"low": 1,  "medium": 2, "high": 3}
DIFFICULTY_MAP    = {"easy": 1, "moderate": 2, "hard": 3}
_EPSILON = 1e-6

CATEGORY_OPTIONS = {
    "gender":          ["female", "male", "other"],
    "course":          ["b.com", "b.sc", "b.tech", "ba", "bba", "bca", "diploma"],
    "internet_access": ["yes", "no"],
    "study_method":    ["coaching", "group study", "mixed", "online videos", "self-study"],
    "sleep_quality":   ["poor", "average", "good"],
    "facility_rating": ["low", "medium", "high"],
    "exam_difficulty": ["easy", "moderate", "hard"],
}

def load_artifacts():
    if not MODEL_PATH.exists() or not ENCODERS_PATH.exists():
        return None, None
    model    = joblib.load(MODEL_PATH)
    encoders = joblib.load(ENCODERS_PATH)
    return model, encoders

def feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    d["sleep_quality_encoded"]        = d["sleep_quality"].map(SLEEP_QUALITY_MAP).fillna(0)
    d["facility_rating_encoded"]      = d["facility_rating"].map(FACILITY_MAP).fillna(0)
    d["exam_difficulty_encoded"]      = d["exam_difficulty"].map(DIFFICULTY_MAP).fillna(0)
    d["study_efficiency"]             = d["study_hours"] * (d["class_attendance"] / 100.0)
    d["sleep_study_ratio"]            = d["sleep_hours"] / (d["study_hours"] + _EPSILON)
    d["engagement_score"]             = (d["class_attendance"] + d["study_hours"] * 10) / 2
    d["study_hours_squared"]          = d["study_hours"] ** 2
    d["class_attendance_squared"]     = d["class_attendance"] ** 2
    d["overall_performance"]          = (
        d["study_efficiency"]
        + 2 * d["sleep_quality_encoded"]
        + 2 * d["facility_rating_encoded"]
        - d["exam_difficulty_encoded"]
    )
    d["study_sleep_balance"]          = np.abs(d["study_hours"] - d["sleep_hours"])
    d["effort_score"]                 = (
        0.6 * d["study_hours"]
        + 0.3 * d["class_attendance"]
        + 0.1 * d["sleep_quality_encoded"] * 10
    )
    d["attendance_study_interaction"] = d["class_attendance"] * d["study_hours"]
    d["difficulty_adjusted_effort"]   = d["effort_score"] / (d["exam_difficulty_encoded"] + 1)
    return d

def prepare_input(payload: dict, encoders: dict) -> pd.DataFrame:
    df = pd.DataFrame([payload])
    df = feature_engineering(df)

    course_enc       = encoders.get("course_encoder", {})
    study_method_enc = encoders.get("study_method_encoder", {})
    gender_cols      = encoders.get("gender_columns", [])
    feat_cols        = encoders.get("feature_columns", [])

    course_mean = float(np.mean(list(course_enc.values()))) if course_enc else 0.0
    method_mean = float(np.mean(list(study_method_enc.values()))) if study_method_enc else 0.0
    df["course_encoded"]       = df["course"].map(course_enc).fillna(course_mean)
    df["study_method_encoded"] = df["study_method"].map(study_method_enc).fillna(method_mean)

    for col in gender_cols:
        df[col] = (df["gender"] == col.replace("gender_", "")).astype(int)

    df["internet_access_binary"] = df["internet_access"].map({"yes": 1, "no": 0}).fillna(0)
    df = df.reindex(columns=feat_cols, fill_value=0)
    return df.astype(np.float32)

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

def create_feature_importance(model, encoders: dict):
    if not hasattr(model, "feature_importances_"):
        return None
    feat_cols = encoders.get("feature_columns", [])
    if not feat_cols:
        return None
    imp_df = pd.DataFrame({
        "feature":    feat_cols,
        "importance": model.feature_importances_,
    }).sort_values("importance", ascending=True).tail(15)
    fig = px.bar(
        imp_df, x="importance", y="feature", orientation="h",
        title="Top 15 Feature Importance",
        color="importance", color_continuous_scale="Blues",
    )
    fig.update_layout(
        height=450, showlegend=False,
        coloraxis_showscale=False,
        margin=dict(l=10, r=10, t=50, b=10),
    )
    return fig

def get_model_params(model) -> dict:
    try:
        return model.get_params()
    except Exception:
        pass
    try:
        booster  = model.get_booster()
        cfg      = json.loads(booster.save_config())
        tree_cfg = (cfg.get("learner", {})
                       .get("gradient_booster", {})
                       .get("tree_train_param", {}))
        return {
            "n_estimators":  booster.num_boosted_rounds(),
            "max_depth":     tree_cfg.get("max_depth", "—"),
            "learning_rate": (cfg.get("learner", {})
                                 .get("learner_train_param", {})
                                 .get("learning_rate", "—")),
            "subsample":     tree_cfg.get("subsample", "—"),
        }
    except Exception:
        return {}

def build_sidebar() -> dict:
    with st.sidebar:
        st.title("🎓 Student Score Predictor")
        st.markdown("---")
        st.markdown("### 👤 Öğrenci Bilgileri")
        age              = st.number_input("Age",                     17,   24,  20)
        gender           = st.selectbox("Gender",            CATEGORY_OPTIONS["gender"])
        course           = st.selectbox("Course",            CATEGORY_OPTIONS["course"])
        study_hours      = st.slider("Study Hours / Day",    0.0,  10.0,  4.0, 0.1)
        class_attendance = st.slider("Class Attendance (%)", 40.0, 99.4, 72.0, 0.5)
        sleep_hours      = st.slider("Sleep Hours / Day",    4.1,   9.9,  7.0, 0.1)
        st.markdown("### 📋 Diğer Bilgiler")
        sleep_quality   = st.selectbox("Sleep Quality",     CATEGORY_OPTIONS["sleep_quality"])
        internet_access = st.selectbox("Internet Access",   CATEGORY_OPTIONS["internet_access"])
        study_method    = st.selectbox("Study Method",      CATEGORY_OPTIONS["study_method"])
        facility_rating = st.selectbox("Facility Rating",   CATEGORY_OPTIONS["facility_rating"])
        exam_difficulty = st.selectbox("Exam Difficulty",   CATEGORY_OPTIONS["exam_difficulty"])
        st.markdown("---")
        predict_clicked = st.button(
            "🔍 Predict Score", use_container_width=True, type="primary"
        )

    return {
        "payload": {
            "age": age, "gender": gender, "course": course,
            "study_hours": study_hours, "class_attendance": class_attendance,
            "sleep_hours": sleep_hours, "sleep_quality": sleep_quality,
            "internet_access": internet_access, "study_method": study_method,
            "facility_rating": facility_rating, "exam_difficulty": exam_difficulty,
        },
        "predict_clicked": predict_clicked,
    }

def main():
    model, encoders = load_artifacts()
    sidebar         = build_sidebar()
    payload         = sidebar["payload"]
    predict_clicked = sidebar["predict_clicked"]

    if model is None or encoders is None:
        st.error(
            "⚠️ **Model dosyaları bulunamadı!**\n\n"
            "`models/xgb_model.joblib` ve `models/encoders.joblib` "
            "dosyalarının mevcut olduğundan emin ol.\n\n"
            "```bash\npython save_model.py\n```"
        )
        st.stop()

    features = prepare_input(payload, encoders)
    score    = float(np.clip(model.predict(features)[0], 0, 100))
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
                    "Yaş", "Cinsiyet", "Bölüm", "Günlük Ders Saati",
                    "Devam Oranı", "Uyku Saati", "Uyku Kalitesi",
                    "İnternet", "Çalışma Yöntemi", "Tesis Kalitesi", "Sınav Zorluğu",
                ],
                "Değer": [
                    payload["age"],              payload["gender"],
                    payload["course"],           f"{payload['study_hours']} saat",
                    f"%{payload['class_attendance']}",
                    f"{payload['sleep_hours']} saat",
                    payload["sleep_quality"],    payload["internet_access"],
                    payload["study_method"],     payload["facility_rating"],
                    payload["exam_difficulty"],
                ],
            }),
            use_container_width=True,
            hide_index=True,
        )

    st.markdown("---")

    st.markdown("#### 🔬 Feature Importance")
    fig_imp = create_feature_importance(model, encoders)
    if fig_imp:
        st.plotly_chart(fig_imp, use_container_width=True)

    with st.expander("⚙️ Model Parametreleri"):
        params = get_model_params(model)
        if params:
            p1, p2, p3, p4 = st.columns(4)
            p1.metric("n_estimators",  params.get("n_estimators",  "—"))
            p2.metric("max_depth",     params.get("max_depth",     "—"))
            p3.metric("learning_rate", params.get("learning_rate", "—"))
            p4.metric("subsample",     params.get("subsample",     "—"))
        st.caption("Model tipi: XGBoost Regressor")

    st.markdown("---")
    st.caption(
        "Powered by XGBoost & Streamlit | "
        "[GitHub](https://github.com/tugcesi/predicting-student-test-scores)"
    )

if __name__ == "__main__":
    main()
"""Streamlit app for student exam score prediction."""

from __future__ import annotations

from pathlib import Path
from zipfile import ZipFile

import joblib
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from config import CATEGORICAL_FEATURES, CATEGORY_OPTIONS, ENCODERS_PATH, MODEL_PATH
from data_processing import feature_engineering, prepare_input

APP_TITLE = "🎓 Student Exam Score Prediction"


@st.cache_resource
def load_artifacts():
    if not MODEL_PATH.exists() or not ENCODERS_PATH.exists():
        return None, None
    model = joblib.load(MODEL_PATH)
    encoders = joblib.load(ENCODERS_PATH)
    return model, encoders


@st.cache_data
def load_training_data():
    data_file = Path("playground-series-s6e1.zip")
    if not data_file.exists():
        return None
    with ZipFile(data_file) as zip_ref, zip_ref.open("train.csv") as csv_file:
        return pd.read_csv(csv_file)


def classify_performance(score: float) -> str:
    if score >= 90:
        return "Excellent"
    if score >= 80:
        return "Very Good"
    if score >= 70:
        return "Good"
    if score >= 60:
        return "Average"
    return "Poor"


def assign_grade(score: float) -> str:
    if score >= 90:
        return "A"
    if score >= 80:
        return "B"
    if score >= 70:
        return "C"
    if score >= 60:
        return "D"
    return "F"


def create_gauge(score: float):
    return go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=score,
            number={"suffix": " / 100"},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": "#1f77b4"},
                "steps": [
                    {"range": [0, 60], "color": "#ffdddd"},
                    {"range": [60, 70], "color": "#ffe9c7"},
                    {"range": [70, 80], "color": "#fff7b2"},
                    {"range": [80, 90], "color": "#d9f7be"},
                    {"range": [90, 100], "color": "#b7eb8f"},
                ],
            },
            title={"text": "Predicted Exam Score"},
        )
    )


def build_input_payload():
    age = st.number_input("Age", min_value=17, max_value=24, value=20)
    gender = st.selectbox("Gender", CATEGORY_OPTIONS["gender"])
    course = st.selectbox("Course", CATEGORY_OPTIONS["course"])
    study_hours = st.slider("Study Hours", min_value=0.0, max_value=10.0, value=4.0, step=0.1)
    class_attendance = st.slider("Class Attendance (%)", min_value=0.0, max_value=100.0, value=80.0, step=0.1)
    sleep_hours = st.slider("Sleep Hours", min_value=0.0, max_value=12.0, value=7.0, step=0.1)
    sleep_quality = st.selectbox("Sleep Quality", CATEGORY_OPTIONS["sleep_quality"])
    internet_access = st.selectbox("Internet Access", CATEGORY_OPTIONS["internet_access"])
    study_method = st.selectbox("Study Method", CATEGORY_OPTIONS["study_method"])
    facility_rating = st.selectbox("Facility Rating", CATEGORY_OPTIONS["facility_rating"])
    exam_difficulty = st.selectbox("Exam Difficulty", CATEGORY_OPTIONS["exam_difficulty"])

    return {
        "age": age,
        "gender": gender,
        "course": course,
        "study_hours": study_hours,
        "class_attendance": class_attendance,
        "sleep_hours": sleep_hours,
        "sleep_quality": sleep_quality,
        "internet_access": internet_access,
        "study_method": study_method,
        "facility_rating": facility_rating,
        "exam_difficulty": exam_difficulty,
    }


def prediction_view(model, encoders):
    st.subheader("🔮 Tahmin Yap")
    st.write("Fill in student details and click **Predict Score**.")

    payload = build_input_payload()
    predict_clicked = st.button("Predict Score")

    if model is None or encoders is None:
        st.warning("Model files not found. Run `python train_model.py` first.")
        return

    features = prepare_input(payload, encoders)
    score = float(model.predict(features)[0])
    score = max(0.0, min(100.0, score))

    if predict_clicked:
        st.success("Prediction generated successfully.")

    st.metric("Predicted Score", f"{score:.2f}")
    st.metric("Performance", classify_performance(score))
    st.metric("Grade", assign_grade(score))
    st.plotly_chart(create_gauge(score), use_container_width=True)


def model_stats_view(model, encoders):
    st.subheader("📊 Model İstatistikleri")

    if model is None or encoders is None:
        st.warning("Model files not found. Run `python train_model.py` first.")
        return

    params = model.get_params()
    st.markdown(
        f"""
        - **Model**: XGBoost Regressor
        - **n_estimators**: {params.get('n_estimators')}
        - **max_depth**: {params.get('max_depth')}
        - **learning_rate**: {params.get('learning_rate')}
        - **subsample**: {params.get('subsample')}
        """
    )

    if hasattr(model, "feature_importances_"):
        importance_df = pd.DataFrame(
            {
                "feature": encoders["feature_columns"],
                "importance": model.feature_importances_,
            }
        ).sort_values("importance", ascending=False)

        st.plotly_chart(
            px.bar(
                importance_df.head(20),
                x="importance",
                y="feature",
                orientation="h",
                title="Top 20 Feature Importance",
            ),
            use_container_width=True,
        )

    data = load_training_data()
    if data is not None:
        sample = data.sample(n=min(1000, len(data)), random_state=42)
        sample_features = sample.drop(columns=["exam_score", "id"], errors="ignore")
        sample_features = feature_engineering(sample_features)
        sample_features = pd.get_dummies(sample_features, columns=CATEGORICAL_FEATURES, drop_first=False)
        sample_features = sample_features.reindex(columns=encoders["feature_columns"], fill_value=0)
        predictions = model.predict(sample_features)
        st.plotly_chart(
            px.histogram(
                pd.DataFrame({"predicted_score": predictions}),
                x="predicted_score",
                nbins=30,
                title="Prediction Distribution",
            ),
            use_container_width=True,
        )


def about_view():
    st.subheader("ℹ️ Hakkında")
    st.markdown(
        """
        This Streamlit app predicts student exam scores using an XGBoost regression model.

        ### How to use
        1. Train the model with `python train_model.py`.
        2. Open this app with `streamlit run app.py`.
        3. Enter student details in **🔮 Tahmin Yap**.
        4. Review model insights in **📊 Model İstatistikleri**.

        The app returns:
        - Predicted score (0-100)
        - Performance class (Excellent/Very Good/Good/Average/Poor)
        - Letter grade (A/B/C/D/F)
        """
    )


def main():
    st.set_page_config(page_title=APP_TITLE, page_icon="🎓", layout="wide")
    st.title(APP_TITLE)

    model, encoders = load_artifacts()

    st.sidebar.title("Sections")
    section = st.sidebar.radio(
        "Navigate",
        ["🔮 Tahmin Yap", "📊 Model İstatistikleri", "ℹ️ Hakkında"],
    )

    if section == "🔮 Tahmin Yap":
        prediction_view(model, encoders)
    elif section == "📊 Model İstatistikleri":
        model_stats_view(model, encoders)
    else:
        about_view()


if __name__ == "__main__":
    main()

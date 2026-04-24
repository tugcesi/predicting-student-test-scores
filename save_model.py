"""
Çalıştır: python save_model.py
Çıktı: models/xgb_model.joblib + models/encoders.joblib
"""

from __future__ import annotations
import os
from zipfile import ZipFile

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor

TARGET    = "exam_score"
MODEL_DIR = "models"

SLEEP_QUALITY_MAP = {"poor": 1, "average": 2, "good": 3}
FACILITY_MAP      = {"low": 1,  "medium": 2, "high": 3}
DIFFICULTY_MAP    = {"easy": 1, "moderate": 2, "hard": 3}
_EPSILON = 1e-6

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

def main():
    # Veri yükle
    if os.path.exists("train.csv"):
        data = pd.read_csv("train.csv")
    elif os.path.exists("playground-series-s6e1.zip"):
        with ZipFile("playground-series-s6e1.zip") as z, z.open("train.csv") as f:
            data = pd.read_csv(f)
    else:
        raise FileNotFoundError("train.csv veya playground-series-s6e1.zip bulunamadı.")

    print(f"Veri: {data.shape}")
    data = data.dropna(subset=[TARGET]).copy()

    # Feature engineering
    data = feature_engineering(data)

    # Target encoding
    course_encoder       = data.groupby("course")[TARGET].mean().to_dict()
    study_method_encoder = data.groupby("study_method")[TARGET].mean().to_dict()
    data["course_encoded"]       = data["course"].map(course_encoder)
    data["study_method_encoded"] = data["study_method"].map(study_method_encoder)

    # Gender one-hot
    gender_dummies = pd.get_dummies(data["gender"], prefix="gender")
    gender_columns = list(gender_dummies.columns)
    data           = pd.concat([data, gender_dummies], axis=1)

    # Internet binary
    data["internet_access_binary"] = data["internet_access"].map({"yes": 1, "no": 0}).fillna(0)

    # Feature listesi
    NUMERIC_FEATURES    = ["age", "study_hours", "class_attendance", "sleep_hours"]
    ENGINEERED_FEATURES = [
        "sleep_quality_encoded", "facility_rating_encoded", "exam_difficulty_encoded",
        "study_efficiency", "sleep_study_ratio", "engagement_score",
        "study_hours_squared", "class_attendance_squared", "overall_performance",
        "study_sleep_balance", "effort_score", "attendance_study_interaction",
        "difficulty_adjusted_effort",
    ]
    final_features = (
        NUMERIC_FEATURES
        + ENGINEERED_FEATURES
        + ["internet_access_binary", "course_encoded", "study_method_encoded"]
        + gender_columns
    )

    X = data[final_features].astype(np.float32)
    y = data[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Model eğit
    print("XGBoost eğitiliyor...")
    model = XGBRegressor(n_estimators=200, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    print(f"R2:   {r2_score(y_test, preds):.4f}")
    print(f"RMSE: {mean_squared_error(y_test, preds)**0.5:.4f}")
    print(f"MAE:  {mean_absolute_error(y_test, preds):.4f}")

    # Kaydet
    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(model, f"{MODEL_DIR}/xgb_model.joblib")
    joblib.dump(
        {
            "course_encoder":       course_encoder,
            "study_method_encoder": study_method_encoder,
            "gender_columns":       gender_columns,
            "feature_columns":      final_features,
        },
        f"{MODEL_DIR}/encoders.joblib",
    )

    print("\nKaydedilen dosyalar:")
    print(f"   models/xgb_model.joblib  ({os.path.getsize(f'{MODEL_DIR}/xgb_model.joblib')//1024} KB)")
    print(f"   models/encoders.joblib   ({os.path.getsize(f'{MODEL_DIR}/encoders.joblib')//1024} KB)")

    # Dogrula
    enc = joblib.load(f"{MODEL_DIR}/encoders.joblib")
    print("\nEncoder keys:", list(enc.keys()))
    print("   course_encoder ornek:", dict(list(enc["course_encoder"].items())[:3]))
    print("   gender_columns:", enc["gender_columns"])

if __name__ == "__main__":
    main()
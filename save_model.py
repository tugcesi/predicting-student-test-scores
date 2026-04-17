"""Train and save the XGBoost model artifacts used by the Streamlit app."""

from __future__ import annotations

import os
from zipfile import ZipFile

import joblib
import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor

from config import ENGINEERED_FEATURES, NUMERIC_FEATURES
from data_processing import feature_engineering

TARGET_COLUMN = "exam_score"
MODEL_DIR = "models"
MODEL_FILE = os.path.join(MODEL_DIR, "xgb_model.joblib")
ENCODERS_FILE = os.path.join(MODEL_DIR, "encoders.joblib")
FEATURES_FILE = os.path.join(MODEL_DIR, "features.joblib")


def _format_size(path: str) -> str:
    size_kb = os.path.getsize(path) / 1024
    return f"{size_kb:.2f} KB"


def main() -> None:
    print("📥 Loading train.csv...")
    if os.path.exists("train.csv"):
        data = pd.read_csv("train.csv")
    elif os.path.exists("data/train.csv"):
        data = pd.read_csv("data/train.csv")
    elif os.path.exists("playground-series-s6e1.zip"):
        with ZipFile("playground-series-s6e1.zip") as zip_ref, zip_ref.open("train.csv") as csv_file:
            data = pd.read_csv(csv_file)
    else:
        raise FileNotFoundError(
            "train.csv not found. Place it in project root, data/ folder, or provide playground-series-s6e1.zip."
        )

    print(f"📊 Raw data shape: {data.shape}")
    data = data.dropna(subset=[TARGET_COLUMN]).copy()
    print(f"🧹 Data shape after dropping null {TARGET_COLUMN}: {data.shape}")

    print("🛠️ Applying feature engineering...")
    data = feature_engineering(data)

    print("🎯 Applying target encoding (course, study_method)...")
    course_encoder = data.groupby("course")[TARGET_COLUMN].mean().to_dict()
    study_method_encoder = data.groupby("study_method")[TARGET_COLUMN].mean().to_dict()
    data["course_encoded"] = data["course"].map(course_encoder)
    data["study_method_encoded"] = data["study_method"].map(study_method_encoder)

    print("🧩 Applying one-hot encoding (gender)...")
    gender_encoder = preprocessing.OneHotEncoder(sparse_output=False, handle_unknown="ignore")
    gender_matrix = gender_encoder.fit_transform(data[["gender"]])
    gender_columns = [f"gender_{value}" for value in gender_encoder.categories_[0]]
    gender_df = pd.DataFrame(gender_matrix, columns=gender_columns, index=data.index)
    data = pd.concat([data, gender_df], axis=1)

    data["internet_access_binary"] = data["internet_access"].map({"yes": 1, "no": 0}).fillna(0)

    final_features = (
        NUMERIC_FEATURES
        + ENGINEERED_FEATURES
        + ["internet_access_binary", "course_encoded", "study_method_encoded"]
        + gender_columns
    )
    X = data[final_features].astype(np.float32)
    y = data[TARGET_COLUMN]

    print(f"🧠 Final training feature shape: {X.shape}")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print("🤖 Training XGBoost model (n_estimators=200)...")
    model = XGBRegressor(n_estimators=200, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    r2 = r2_score(y_test, predictions)
    rmse = mean_squared_error(y_test, predictions) ** 0.5
    mae = mean_absolute_error(y_test, predictions)

    baseline_model = LinearRegression()
    baseline_model.fit(X_train, y_train)
    baseline_r2 = r2_score(y_test, baseline_model.predict(X_test))

    print("📈 Model performance:")
    print(f"   • R²   : {r2:.4f}")
    print(f"   • RMSE : {rmse:.4f}")
    print(f"   • MAE  : {mae:.4f}")
    print(f"   • Baseline Linear R²: {baseline_r2:.4f}")

    print("💾 Saving model artifacts...")
    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(model, MODEL_FILE)
    joblib.dump(
        {
            "course_encoder": course_encoder,
            "study_method_encoder": study_method_encoder,
            "gender_columns": gender_columns,
            "feature_columns": final_features,
        },
        ENCODERS_FILE,
    )
    joblib.dump(final_features, FEATURES_FILE)

    print("✅ Saved files:")
    print(f"   • {MODEL_FILE} ({_format_size(MODEL_FILE)})")
    print(f"   • {ENCODERS_FILE} ({_format_size(ENCODERS_FILE)})")
    print(f"   • {FEATURES_FILE} ({_format_size(FEATURES_FILE)})")
    print("🚀 Next steps:")
    print("   1) Run: python save_model.py (if you update data)")
    print("   2) Run: streamlit run app.py")


if __name__ == "__main__":
    main()

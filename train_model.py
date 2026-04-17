"""Train and evaluate the XGBoost model for exam score prediction."""

from __future__ import annotations

from pathlib import Path
from zipfile import ZipFile

import joblib
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score, train_test_split
from xgboost import XGBRegressor

from config import CATEGORICAL_FEATURES, ENCODERS_PATH, MODEL_DIR, MODEL_PATH
from data_processing import feature_engineering

TARGET_COLUMN = "exam_score"


def _read_training_data(data_path: str | Path) -> pd.DataFrame:
    data_path = Path(data_path)

    if data_path.suffix.lower() == ".zip":
        with ZipFile(data_path) as zip_ref, zip_ref.open("train.csv") as csv_file:
            return pd.read_csv(csv_file)

    return pd.read_csv(data_path)


def load_and_prepare_data(data_path: str | Path = "playground-series-s6e1.zip"):
    """Load dataset and transform it into model-ready features."""
    data = _read_training_data(data_path)
    data = feature_engineering(data)

    X = data.drop(columns=[TARGET_COLUMN, "id"], errors="ignore")
    X = pd.get_dummies(X, columns=CATEGORICAL_FEATURES, drop_first=False)
    y = data[TARGET_COLUMN]

    return X, y


def evaluate_model(model, X: pd.DataFrame, y: pd.Series, X_test: pd.DataFrame, y_test: pd.Series):
    """Compute model quality metrics."""
    predictions = model.predict(X_test)

    metrics = {
        "r2": r2_score(y_test, predictions),
        "rmse": mean_squared_error(y_test, predictions) ** 0.5,
        "mae": mean_absolute_error(y_test, predictions),
    }

    cv_scores = cross_val_score(model, X, y, cv=5, scoring="r2")
    metrics["cv_scores"] = cv_scores.tolist()
    metrics["cv_mean"] = cv_scores.mean()
    metrics["cv_std"] = cv_scores.std()

    return metrics


def train_and_save_model(data_path: str | Path = "playground-series-s6e1.zip"):
    """Train XGBoost model, evaluate it, and save artifacts."""
    X, y = load_and_prepare_data(data_path)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
    )

    model = XGBRegressor(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.9,
        colsample_bytree=0.9,
        random_state=42,
        n_jobs=-1,
    )

    model.fit(X_train, y_train)
    metrics = evaluate_model(model, X, y, X_test, y_test)

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    joblib.dump({"feature_columns": X.columns.tolist()}, ENCODERS_PATH)

    print("Model saved to:", MODEL_PATH)
    print("Encoders saved to:", ENCODERS_PATH)
    print("Metrics:")
    for key, value in metrics.items():
        if isinstance(value, list):
            print(f"- {key}: {[round(v, 4) for v in value]}")
        else:
            print(f"- {key}: {value:.4f}")

    return model, metrics


if __name__ == "__main__":
    train_and_save_model()

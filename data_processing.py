"""Data processing and feature engineering utilities."""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from config import (
    CATEGORICAL_FEATURES,
    DIFFICULTY_MAP,
    FACILITY_MAP,
    FEATURE_RANGES,
    SLEEP_QUALITY_MAP,
    CATEGORY_OPTIONS,
)

_EPSILON = 1e-6


def feature_engineering(data: pd.DataFrame) -> pd.DataFrame:
    """Create engineered features used by the model."""
    df = data.copy()

    df["sleep_quality_encoded"] = df["sleep_quality"].map(SLEEP_QUALITY_MAP).fillna(0)
    df["facility_rating_encoded"] = df["facility_rating"].map(FACILITY_MAP).fillna(0)
    df["exam_difficulty_encoded"] = df["exam_difficulty"].map(DIFFICULTY_MAP).fillna(0)

    df["study_efficiency"] = df["study_hours"] * (df["class_attendance"] / 100.0)
    df["sleep_study_ratio"] = df["sleep_hours"] / (df["study_hours"] + _EPSILON)
    df["engagement_score"] = (df["class_attendance"] + df["study_hours"] * 10) / 2
    df["study_hours_squared"] = df["study_hours"] ** 2
    df["class_attendance_squared"] = df["class_attendance"] ** 2
    df["overall_performance"] = (
        df["study_efficiency"]
        + 2 * df["sleep_quality_encoded"]
        + 2 * df["facility_rating_encoded"]
        - df["exam_difficulty_encoded"]
    )
    df["study_sleep_balance"] = np.abs(df["study_hours"] - df["sleep_hours"])
    df["effort_score"] = (
        0.6 * df["study_hours"]
        + 0.3 * df["class_attendance"]
        + 0.1 * df["sleep_quality_encoded"] * 10
    )
    df["attendance_study_interaction"] = df["class_attendance"] * df["study_hours"]
    df["difficulty_adjusted_effort"] = df["effort_score"] / (df["exam_difficulty_encoded"] + 1)

    return df


def validate_input(user_input: Dict[str, object]) -> Tuple[bool, List[str]]:
    """Validate raw user input values for prediction."""
    errors: List[str] = []

    for feature, (min_value, max_value) in FEATURE_RANGES.items():
        value = user_input.get(feature)
        if value is None:
            errors.append(f"Missing value for '{feature}'.")
            continue
        if not (min_value <= float(value) <= max_value):
            errors.append(f"'{feature}' should be between {min_value} and {max_value}.")

    for feature, options in CATEGORY_OPTIONS.items():
        value = user_input.get(feature)
        if value is None:
            errors.append(f"Missing value for '{feature}'.")
            continue
        if value not in options:
            errors.append(f"'{feature}' must be one of: {', '.join(options)}.")

    return len(errors) == 0, errors


def prepare_input(user_input: Dict[str, object], encoder_data: Dict[str, object]) -> pd.DataFrame:
    """Prepare a single user input payload to model-ready features."""
    is_valid, errors = validate_input(user_input)
    if not is_valid:
        raise ValueError("Invalid user input: " + " ".join(errors))

    input_df = pd.DataFrame([user_input])
    input_df = feature_engineering(input_df)
    input_df = pd.get_dummies(input_df, columns=CATEGORICAL_FEATURES, drop_first=False)

    feature_columns = encoder_data["feature_columns"]
    input_df = input_df.reindex(columns=feature_columns, fill_value=0)

    return input_df

"""Project configuration for student exam score prediction."""

from pathlib import Path

# Core feature groups
NUMERIC_FEATURES = [
    "age",
    "study_hours",
    "class_attendance",
    "sleep_hours",
]

CATEGORICAL_FEATURES = [
    "gender",
    "course",
    "internet_access",
    "study_method",
    "sleep_quality",
    "facility_rating",
    "exam_difficulty",
]

ENGINEERED_FEATURES = [
    "study_efficiency",
    "sleep_study_ratio",
    "engagement_score",
    "study_hours_squared",
    "class_attendance_squared",
    "sleep_quality_encoded",
    "facility_rating_encoded",
    "exam_difficulty_encoded",
    "overall_performance",
    "study_sleep_balance",
    "effort_score",
    "attendance_study_interaction",
    "difficulty_adjusted_effort",
]

# Ordinal encoding maps
SLEEP_QUALITY_MAP = {"poor": 1, "average": 2, "good": 3}
FACILITY_MAP = {"low": 1, "medium": 2, "high": 3}
DIFFICULTY_MAP = {"easy": 1, "moderate": 2, "hard": 3}

# Input validation ranges
FEATURE_RANGES = {
    "age": (17, 24),
    "study_hours": (0.0, 10.0),
    "class_attendance": (0.0, 100.0),
    "sleep_hours": (0.0, 12.0),
}

# Dropdown options
CATEGORY_OPTIONS = {
    "gender": ["female", "male", "other"],
    "course": ["b.com", "b.sc", "b.tech", "ba", "bba", "bca", "diploma"],
    "internet_access": ["yes", "no"],
    "study_method": ["coaching", "group study", "mixed", "online videos", "self-study"],
    "sleep_quality": ["poor", "average", "good"],
    "facility_rating": ["low", "medium", "high"],
    "exam_difficulty": ["easy", "moderate", "hard"],
}

# Model artifact paths
BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models"
MODEL_PATH = MODEL_DIR / "xgb_model.joblib"
ENCODERS_PATH = MODEL_DIR / "encoders.joblib"

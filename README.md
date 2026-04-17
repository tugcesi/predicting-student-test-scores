# Predicting Student Test Scores

A machine learning project that predicts student exam scores from study habits, attendance, and learning environment features. The project includes data preparation, XGBoost model training, and a Streamlit web app for interactive predictions.

## Features

- End-to-end training pipeline with feature engineering
- 12+ engineered features (efficiency, balance, engagement, effort, ordinal encodings)
- XGBoost regressor model export with Joblib
- Streamlit UI with:
  - score prediction
  - performance class (Excellent/Very Good/Good/Average/Poor)
  - letter grade (A/B/C/D/F)
  - Plotly gauge visualization
  - model insights and feature importance

## Requirements

- Python 3.10+
- Dependencies listed in `requirements.txt`

Install dependencies:

```bash
pip install -r requirements.txt
```

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/tugcesi/predicting-student-test-scores.git
   cd predicting-student-test-scores
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Train and save model artifacts:
   ```bash
   python train_model.py
   ```

## Usage

Run the Streamlit application:

```bash
streamlit run app.py
```

Then use the sidebar sections:

- **🔮 Tahmin Yap**: Enter student details and generate predictions
- **📊 Model İstatistikleri**: View model parameters and feature importance
- **ℹ️ Hakkında**: Read project information and usage notes

## Model Performance

Current reference performance metrics:

- **R²:** 0.82
- **RMSE:** 12.5
- **MAE:** 9.2

## Project Structure

```text
.
├── app.py
├── config.py
├── data_processing.py
├── train_model.py
├── requirements.txt
├── models/
│   ├── xgb_model.joblib
│   └── encoders.joblib
└── README.md
```

## Technical Details

- **Model:** `XGBRegressor` (`n_estimators=200`)
- **Target:** `exam_score`
- **Categorical handling:** One-hot encoding
- **Ordinal features:** Sleep quality, facility rating, exam difficulty
- **Validation metrics:** R², RMSE, MAE, and 5-fold cross-validation scores

## Author

Tugce Basyigit

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE).

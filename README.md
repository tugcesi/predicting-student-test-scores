# 🎓 Predicting Student Test Scores

A deep learning project that predicts student exam scores from study habits, attendance, and learning environment features. The project includes exploratory data analysis, a Keras neural network model, and a Streamlit web application.

## ✨ Features

- Deep learning model built with **TensorFlow/Keras** (5 Dense layers)
- Input preprocessing hardcoded into the app — **no extra files needed**
- StandardScaler normalization baked in (no `.pkl` or `.joblib`)
- Streamlit UI with:
  - Real-time score prediction
  - Performance class (Excellent / Very Good / Good / Average / Poor)
  - Letter grade (A / B / C / D / F)
  - Plotly gauge visualization
  - Input summary table

## 📋 Requirements

- Python 3.10+
- Dependencies listed in `requirements.txt`

```bash
pip install -r requirements.txt
```

## 🚀 Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/tugcesi/predicting-student-test-scores.git
   cd predicting-student-test-scores
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Place the trained model file:
   ```
   models/testscore_model.keras
   ```

## 🖥️ Usage

Run the Streamlit application:

```bash
streamlit run app.py
```

Use the sidebar to enter student details and get an instant prediction.

## 📊 Model Performance

Training results (10 epochs, EarlyStopping):

| Epoch | Train Loss (MSE) | Val Loss (MSE) |
|-------|-----------------|----------------|
| 1     | 100.06          | 81.92          |
| 5     | 80.72           | 82.69          |
| 8     | 80.28           | 79.52          |
| 10    | 80.06           | 79.90          |

- **Final Val MSE:** ~79.5 → **RMSE ≈ 8.9**

## 🗂️ Project Structure

```text
.
├── app.py                          # Streamlit app (DL model)
├── predictingstudenttestscoreswithMLandDL.ipynb  # Notebook (EDA + ML + DL)
├── requirements.txt
├── models/
│   └── testscore_model.keras       # Trained Keras model
└── README.md
```

## ⚙️ Technical Details

| Özellik | Detay |
|---------|-------|
| **Model** | `Sequential` — Dense(80) → Dense(120) → Dense(64) → Dense(30) → Dense(8) → Dense(1) |
| **Aktivasyon** | ReLU (çıkış katmanı lineer) |
| **Loss** | Mean Squared Error |
| **Optimizer** | Adam |
| **Eğitim** | 10 epoch, batch=64, EarlyStopping(patience=5) |
| **Target** | `exam_score` |
| **Dropped features** | `age`, `internet_access`, `id` |
| **Encoding** | `pd.get_dummies(drop_first=True)` |
| **Scaling** | `StandardScaler` (parametreler app.py'ye hardcoded) |

## 👤 Author

Tugce Basyigit

## 📄 License

This project is licensed under the MIT License. See [LICENSE](LICENSE).

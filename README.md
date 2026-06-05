# Multi-Feature Medical Insurance Cost Predictor

An end-to-end Machine Learning web application utilizing Scikit-Learn processing pipelines and Streamlit to predict annual medical insurance expenses based on individual user profiles.

## 🏗️ Architecture & Processing Pipeline
- **Preprocessing Block:** Automated categorical text encoding using `OneHotEncoder(drop='first')` and numerical feature scaling via `StandardScaler`.
- **Model Engine:** Trained using a `RandomForestRegressor` ensemble wrapped inside an end-to-end `Pipeline` object to completely prevent data leakage during runtime inference.

## 🚀 How To Run Locally

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/yourusername/insurance-predictor.git](https://github.com/yourusername/insurance-predictor.git)
   cd insurance-predictor
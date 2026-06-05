# 🏥 Multi-Feature Medical Insurance Cost Predictor

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.35.0-FF4B4B.svg)](https://streamlit.io/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.4.0-F7931E.svg)](https://scikit-learn.org/)

An end-to-end Machine Learning web application utilizing Scikit-Learn data processing pipelines and Streamlit to predict annual medical insurance expenses based on individual user demographic profiles.

🔗 **Live Deployment Link:** [View Live Web Application on Streamlit Cloud](https://your-custom-app-name.streamlit.app)

---

## 🏗️ Architecture & Data Pipeline

This project emphasizes clean code structure and real-world deployment practices over disorganized Jupyter notebooks.

* **Automated Preprocessing Block:** Handles categorical text encoding natively using `OneHotEncoder(drop='first')` and numerical feature scaling via `StandardScaler`.
* **Model Engine:** Trained using a `RandomForestRegressor` ensemble. 
* **Data Leakage Prevention:** The preprocessing blocks and model configuration are combined inside an end-to-end Scikit-Learn `Pipeline` object. This ensures that live data coming from the web UI uses the exact same parsing rules as the training dataset.

---

## 🛠️ Tech Stack

* **Frontend UI:** Streamlit
* **Machine Learning Framework:** Scikit-Learn
* **Data Analytics:** Pandas, NumPy
* **Model Serialization:** Joblib

---

## 📁 Repository Structure

```text
insurance-predictor/
│
├── data/
│   └── insurance.csv                 # Core training dataset from Kaggle
│
├── src/
│   └── train.py                      # Data preprocessing & model training script
│
├── app.py                            # Streamlit interactive web application interface
├── insurance_model_pipeline.pkl      # Serialized model artifact ready for production
└── requirements.txt                  # Application package dependencies
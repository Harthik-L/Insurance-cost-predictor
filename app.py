import streamlit as st
import pandas as pd
import joblib
import os

# Page configuration layout
st.set_page_config(page_title="Health Insurance Predictor", layout="centered")

st.title("🏥 Medical Insurance Cost Predictor")
st.write("Modify the demographics below to calculate estimated annual medical coverage rates instantly.")

# Verify if the pipeline exists before loading frontend elements
if not os.path.exists("insurance_model_pipeline.pkl"):
    st.error("⚠️ Model file not found! Please run 'python src/train.py' in your terminal first to train the model pipeline.")
else:
    # Load the compiled processing pipeline structure
    pipeline = joblib.load("insurance_model_pipeline.pkl")

    # Layout inputs cleanly using side-by-side columns
    col1, col2 = st.columns(2)

    with col1:
        age = st.slider("Age", min_value=18, max_value=100, value=30)
        sex = st.selectbox("Gender", options=["male", "female"])
        bmi = st.number_input("Body Mass Index (BMI)", min_value=10.0, max_value=50.0, value=24.5, step=0.1)

    with col2:
        children = st.spinbox = st.selectbox("Number of Dependents / Children", options=[0, 1, 2, 3, 4, 5])
        smoker = st.radio("Do you smoke regularly?", options=["no", "yes"])
        region = st.selectbox("Geographic Region", options=["southwest", "southeast", "northwest", "northeast"])

    st.markdown("---")

    # Run calculation step when action button is triggered
    if st.button("Calculate Estimated Premium", type="primary"):
        # Format current selections into an exact matching raw schema format
        input_data = pd.DataFrame([{
            'age': age,
            'sex': sex,
            'bmi': bmi,
            'children': children,
            'smoker': smoker,
            'region': region
        }])

        # Predict using the saved pipeline
        prediction = pipeline.predict(input_data)[0]

        # Display output formatting
        st.success(f"### Predicted Annual Insurance Premium: **${prediction:,.2f}**")
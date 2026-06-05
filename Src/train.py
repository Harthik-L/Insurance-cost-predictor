import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
import joblib

def load_or_create_data():
    csv_path = "data/insurance.csv"
    # Auto-generate sample data if the CSV file does not exist yet
    if not os.path.exists(csv_path):
        os.makedirs("data", exist_ok=True)
        np.random.seed(42)
        n_samples = 1000
        mock_data = pd.DataFrame({
            'age': np.random.randint(18, 65, n_samples),
            'sex': np.random.choice(['male', 'female'], n_samples),
            'bmi': np.random.uniform(18.5, 40.0, n_samples),
            'children': np.random.randint(0, 5, n_samples),
            'smoker': np.random.choice(['yes', 'no'], n_samples, p=[0.2, 0.8]),
            'region': np.random.choice(['southwest', 'southeast', 'northwest', 'northeast'], n_samples)
        })
        # Calculate a realistic mock cost structure
        mock_data['charges'] = (mock_data['age'] * 250) + (mock_data['bmi'] * 300) + \
                               (mock_data['smoker'] == 'yes').astype(int) * 15000 + \
                               (mock_data['children'] * 500) + np.random.normal(0, 1000, n_samples)
        mock_data.to_csv(csv_path, index=False)
    return pd.read_csv(csv_path)

def build_and_save_pipeline():
    print("⚡ Loading dataset...")
    df = load_or_create_data()

    # Split features and target column
    X = df.drop(columns=['charges'])
    y = df['charges']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Define features grouped by structural type
    num_features = ['age', 'bmi', 'children']
    cat_features = ['sex', 'smoker', 'region']

    # Build preprocessing blocks
    preprocessor = ColumnTransformer(transformers=[
        ('num', StandardScaler(), num_features),
        ('cat', OneHotEncoder(drop='first', handle_unknown='ignore'), cat_features)
    ])

    # Combine preprocessing steps and model logic into a single executive pipeline
    model_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', RandomForestRegressor(n_estimators=100, random_state=42))
    ])

    print("🏋️ Training the RandomForest model pipeline...")
    model_pipeline.fit(X_train, y_train)

    # Calculate and show benchmark performance metric
    score = model_pipeline.score(X_test, y_test)
    print(print(f"✅ Training completed. Test R² Score: {score:.4f}"))

    # Save the entire system pipeline down to a serialized file format
    joblib.dump(model_pipeline, 'insurance_model_pipeline.pkl')
    print("💾 Model successfully saved as 'insurance_model_pipeline.pkl'")

if __name__ == '__main__':
    build_and_save_pipeline()
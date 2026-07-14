import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler, RobustScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

# Load data
df = pd.read_csv('data/diabetes_prediction_dataset.csv')

# Remove duplicates
df.drop_duplicates(inplace=True)

# Prepare features and target
X = df.drop('diabetes', axis=1)
y = df['diabetes']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Define preprocessing
ohe_cols = ['gender', 'smoking_history']
minmax_cols = ['age']
robust_cols = ['bmi', 'HbA1c_level', 'blood_glucose_level']
passthrough_cols = ['hypertension', 'heart_disease']

preprocessor = ColumnTransformer(transformers=[
    ('ohe', OneHotEncoder(drop='first', handle_unknown='ignore'), ohe_cols),
    ('minmax', MinMaxScaler(), minmax_cols),
    ('robust', RobustScaler(), robust_cols),
    ('passthrough', 'passthrough', passthrough_cols)
])

# Create pipeline
pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', LogisticRegression(max_iter=1000, random_state=42))
])

# Train model
print("Training model...")
pipeline.fit(X_train, y_train)

# Evaluate
train_score = pipeline.score(X_train, y_train)
test_score = pipeline.score(X_test, y_test)
print(f"Train Accuracy: {train_score:.4f}")
print(f"Test Accuracy: {test_score:.4f}")

# Save model
print("Saving model...")
with open('diabetes_predictor_pipeline.pkl', 'wb') as f:
    pickle.dump(pipeline, f)

print("✅ Model saved successfully!")

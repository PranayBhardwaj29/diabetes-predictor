import pickle
import pandas as pd

with open('diabetes_predictor_pipeline.pkl', 'rb') as f:
    pipeline = pickle.load(f)

new_patient = pd.DataFrame({
    'gender': ['Male'],
    'age': [45.0],
    'hypertension': [0],
    'heart_disease': [0],
    'smoking_history': ['never'],
    'bmi': [27.5],
    'HbA1c_level': [6.0],
    'blood_glucose_level': [140]
})

prediction = pipeline.predict(new_patient)
probability = pipeline.predict_proba(new_patient)[0][1]

print(f"Diabetes Prediction: {'Yes' if prediction[0] == 1 else 'No'}")
print(f"Probability: {probability:.2%}")
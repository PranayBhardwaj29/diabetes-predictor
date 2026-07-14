# Diabetes Predictor

A machine learning model to predict diabetes using logistic regression.

## Model Pipeline
- **Preprocessing**: OneHotEncoder for categorical, MinMaxScaler for age, RobustScaler for outlier-prone features
- **Classifier**: Logistic Regression

## Usage
```python
import pickle
pipeline = pickle.load(open('diabetes_predictor_pipeline.pkl', 'rb'))
prediction = pipeline.predict(new_data)

The dataset we are working on today is https://www.kaggle.com/datasets/iammustafatz/diabetes-prediction-dataset

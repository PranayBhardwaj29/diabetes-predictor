import streamlit as st
import pickle
import pandas as pd
import numpy as np


st.set_page_config(
    page_title="Diabetes Predictor",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)


st.markdown("""
    <style>
    .main-header {
        text-align: center;
        color: #1f77b4;
        padding: 20px;
    }
    .prediction-box {
        padding: 20px;
        border-radius: 10px;
        text-align: center;
    }
    .high-risk {
        background-color: #ffcccc;
        color: #cc0000;
    }
    .low-risk {
        background-color: #ccffcc;
        color: #00cc00;
    }
    </style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_model():
    try:
        with open('diabetes_predictor_pipeline.pkl', 'rb') as f:
            pipeline = pickle.load(f)
        return pipeline
    except FileNotFoundError:
        st.error("❌ Model file 'diabetes_predictor_pipeline.pkl' not found!")
        return None

pipeline = load_model()


st.markdown("<h1 class='main-header'>🩺 Diabetes Prediction App</h1>", unsafe_allow_html=True)
st.markdown("---")

st.markdown("""
    ### Welcome to the Diabetes Predictor
    This app uses a machine learning model to predict the likelihood of diabetes based on your health metrics.
    Please enter your health information below to get a prediction.
""")

if pipeline is not None:
    
    tab1, tab2 = st.tabs(["Prediction", "About Model"])
    
    with tab1:
        st.subheader("📋 Enter Your Health Information")
        
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Personal Information**")
            gender = st.selectbox("Gender", ["Male", "Female"])
            age = st.slider("Age", min_value=1, max_value=100, value=35)
            
            st.write("**Lifestyle Factors**")
            smoking_history = st.selectbox(
                "Smoking History",
                ["never", "No Info", "current", "former", "ever", "not current"]
            )
        
        with col2:
            st.write("**Medical Conditions**")
            hypertension = st.radio("Hypertension", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
            heart_disease = st.radio("Heart Disease", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
        
        st.markdown("---")
        st.subheader("📊 Health Metrics")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            bmi = st.slider("BMI (Body Mass Index)", min_value=10.0, max_value=50.0, value=25.0, step=0.1)
        
        with col2:
            hba1c_level = st.slider("HbA1c Level (%)", min_value=3.0, max_value=9.0, value=5.5, step=0.1)
        
        with col3:
            blood_glucose_level = st.slider("Blood Glucose Level (mg/dL)", min_value=70, max_value=300, value=140)
        
        
        if st.button("🔍 Predict Diabetes Risk", use_container_width=True):
            
            
            input_data = pd.DataFrame({
                'gender': [gender],
                'age': [float(age)],
                'hypertension': [hypertension],
                'heart_disease': [heart_disease],
                'smoking_history': [smoking_history],
                'bmi': [float(bmi)],
                'HbA1c_level': [float(hba1c_level)],
                'blood_glucose_level': [blood_glucose_level]
            })
            
            
            prediction = pipeline.predict(input_data)[0]
            probability = pipeline.predict_proba(input_data)[0][1]
            
            st.markdown("---")
            st.subheader("📈 Prediction Results")
            
            
            if prediction == 1:
                st.markdown(
                    f"""
                    <div class='prediction-box high-risk'>
                        <h3>⚠️ High Risk of Diabetes</h3>
                        <p style='font-size: 24px; font-weight: bold;'>{probability:.1%}</p>
                        <p>Based on your health metrics, there's a significant probability of diabetes.</p>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
            else:
                st.markdown(
                    f"""
                    <div class='prediction-box low-risk'>
                        <h3>✅ Low Risk of Diabetes</h3>
                        <p style='font-size: 24px; font-weight: bold;'>{(1-probability):.1%}</p>
                        <p>Based on your health metrics, your diabetes risk is low.</p>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
            
            st.markdown("---")
            
            
            st.subheader("📝 Your Information Summary")
            summary_df = pd.DataFrame({
                'Metric': ['Gender', 'Age', 'Hypertension', 'Heart Disease', 'Smoking History', 'BMI', 'HbA1c Level', 'Blood Glucose'],
                'Value': [gender, age, 'Yes' if hypertension == 1 else 'No', 'Yes' if heart_disease == 1 else 'No', 
                         smoking_history, f'{bmi:.1f}', f'{hba1c_level:.1f}%', f'{blood_glucose_level} mg/dL']
            })
            st.table(summary_df)
            
            
            st.markdown("---")
            st.subheader("💡 Health Recommendations")
            recommendations = []
            
            if bmi > 25:
                recommendations.append("🏃 **Weight Management**: Consider maintaining a healthy weight through exercise and diet")
            if hba1c_level > 6.0:
                recommendations.append("🍎 **Blood Sugar Control**: Monitor your blood sugar levels and consider dietary adjustments")
            if blood_glucose_level > 140:
                recommendations.append("⚕️ **Glucose Monitoring**: Regular glucose monitoring is recommended")
            if hypertension == 1 or heart_disease == 1:
                recommendations.append("❤️ **Cardiovascular Health**: Consult with your healthcare provider about your cardiovascular risk")
            if smoking_history != "never":
                recommendations.append("🚭 **Smoking Cessation**: Consider quitting smoking to reduce health risks")
            
            if recommendations:
                for rec in recommendations:
                    st.info(rec)
            else:
                st.success("✨ Keep maintaining your healthy lifestyle!")
    
    with tab2:
        st.subheader("📚 About the Model")
        st.markdown("""
            ### Model Details
            - **Algorithm**: Logistic Regression
            - **Training Data**: 96,146 samples
            - **Test Set Accuracy**: ~92%
            
            ### Features Used
            1. **Gender**: Male/Female
            2. **Age**: Years (0-100)
            3. **Hypertension**: Presence of hypertension (0 or 1)
            4. **Heart Disease**: Presence of heart disease (0 or 1)
            5. **Smoking History**: never, No Info, current, former, ever, not current
            6. **BMI**: Body Mass Index (10-50)
            7. **HbA1c Level**: Glycated hemoglobin percentage (3-9%)
            8. **Blood Glucose Level**: Fasting blood glucose (70-300 mg/dL)
            
            ### Preprocessing
            - **OneHotEncoding**: Applied to categorical variables (gender, smoking_history)
            - **MinMaxScaler**: Applied to age (0-1 normalization)
            - **RobustScaler**: Applied to numerical features (BMI, HbA1c, Blood Glucose) to handle outliers
            
            ### Important Disclaimer
            ⚠️ **This model is for informational purposes only and should NOT be used as a substitute for professional medical advice.**
            Always consult with a healthcare professional for diagnosis and treatment recommendations.
        """)

else:
    st.error("⚠️ Could not load the model. Please ensure the 'diabetes_predictor_pipeline.pkl' file is in the same directory as app.py")

st.markdown("---")
st.markdown("""
    <div style='text-align: center; color: gray; margin-top: 30px;'>
        <p>💙 Diabetes Predictor | Made with Streamlit</p>
        <p>For medical concerns, always consult a healthcare professional</p>
    </div>
""", unsafe_allow_html=True)
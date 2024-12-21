import streamlit as st
import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import LabelEncoder
import base64
from pathlib import Path

class LabelEncoderTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        self.encoder = LabelEncoder()
        return self

    def transform(self, X):
        return X.apply(self.encoder.fit_transform)

class SleepDisorderApp:
    def __init__(self):
        self.load_models()
        self.setup_constants()
        self.setup_style()

    def load_models(self):
        try:
            self.model = joblib.load('sleep_disorder_random_forest_model.pkl')
            self.preprocessor = joblib.load('preprocessor.pkl')
        except FileNotFoundError as e:
            st.error(f"Error loading model files: {e}")
            st.stop()

    def setup_constants(self):
        self.OCCUPATIONS = ['Others', 'Doctor', 'Teacher', 'Nurse', 'Engineer', 
                           'Accountant', 'Lawyer', 'Salesperson']
        self.BMI_CATEGORIES = ['Normal Weight', 'Overweight', 'Obese']
        self.DISORDER_MAPPING = {
            0.0: ("Insomnia", "red"),
            1.0: ("No Disorder", "green"),
            2.0: ("Sleep Apnea", "orange")
        }

    def setup_style(self):
        try:
            bg_path = Path("Background.jpg")
            if bg_path.exists():
                with open(bg_path, "rb") as image_file:
                    encoded_string = base64.b64encode(image_file.read()).decode()
                self.set_background_style(encoded_string)
            else:
                self.set_default_style()
        except Exception as e:
            st.warning(f"Could not load background image: {e}")
            self.set_default_style()

    def set_background_style(self, encoded_string):
        st.markdown(f"""
        <style>
        body {{
            background-image: url("data:image/jpeg;base64,{encoded_string}");
            background-size: cover;
            background-position: center;
            background-repeat: no-repeat;
        }}
        {self.get_common_styles()}
        </style>
        """, unsafe_allow_html=True)

    def set_default_style(self):
        st.markdown(f"""
        <style>
        body {{
            background-color: #f0f2f6;
        }}
        {self.get_common_styles()}
        </style>
        """, unsafe_allow_html=True)

    def get_common_styles(self):
        return """
        .title {
            font-size: 40px;
            font-weight: bold;
            color: #1e90ff;
            text-align: center;
            margin-bottom: 2rem;
        }
        .stButton>button {
            background-color: #1e90ff;
            color: white;
            font-size: 16px;
            border-radius: 10px;
            width: 100%;
            transition: background-color 0.3s;
        }
        .stButton>button:hover {
            background-color: #4682b4;
        }
        .result-text {
            font-size: 24px;
            font-weight: bold;
            text-align: center;
            margin: 2rem 0;
        }
        .input-section {
            background-color: rgba(255, 255, 255, 0.9);
            padding: 2rem;
            border-radius: 10px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        }
        """

    def preprocess_input(self, data):
        try:
            data['Sleep Duration Sqrt'] = np.sqrt(data['Sleep Duration'])
            data = data.drop(columns=['Sleep Duration'])
            return self.preprocessor.transform(data)
        except Exception as e:
            st.error(f"Error in preprocessing: {e}")
            return None

    def get_user_input(self):
        with st.form(key='input_form'):
            col1, col2 = st.columns(2)
            
            with col1:
                gender = st.selectbox('Gender', ['Male', 'Female'])
                age = st.number_input('Age', min_value=18, max_value=100, value=30,
                                    help="Enter your age (18-100 years)")
                sleep_duration = st.number_input('Sleep Duration (hours)', 
                                               min_value=0.0, max_value=24.0, value=7.5,
                                               help="Average sleep duration in hours")
                occupation = st.selectbox('Occupation', self.OCCUPATIONS)
                quality_of_sleep = st.slider('Quality of Sleep', 1, 10, 7,
                                           help="Rate your sleep quality (1-10)")
                physical_activity = st.number_input('Physical Activity (minutes/day)',
                                                  min_value=0, value=30,
                                                  help="Daily physical activity duration")

            with col2:
                stress_level = st.slider('Stress Level', 1, 10, 5,
                                       help="Rate your stress level (1-10)")
                bmi_category = st.selectbox('BMI Category', self.BMI_CATEGORIES)
                systolic = st.number_input('Systolic Blood Pressure',
                                         min_value=70, max_value=200, value=120,
                                         help="Upper blood pressure number")
                diastolic = st.number_input('Diastolic Blood Pressure',
                                          min_value=40, max_value=130, value=80,
                                          help="Lower blood pressure number")
                heart_rate = st.number_input('Heart Rate (bpm)',
                                           min_value=40, max_value=200, value=70,
                                           help="Resting heart rate in beats per minute")
                daily_steps = st.number_input('Daily Steps',
                                            min_value=0, max_value=50000, value=10000,
                                            help="Average number of steps per day")

            submit_button = st.form_submit_button(label='Predict Sleep Disorder')
            
            return submit_button, {
                'Gender': [gender],
                'Age': [age],
                'Sleep Duration': [sleep_duration],
                'Occupation': [occupation],
                'Quality of Sleep': [quality_of_sleep],
                'Physical Activity Level': [physical_activity],
                'Stress Level': [stress_level],
                'BMI Category': [bmi_category],
                'Systolic': [systolic],
                'Diastolic': [diastolic],
                'Heart Rate': [heart_rate],
                'Daily Steps': [daily_steps]
            }

    def display_prediction(self, prediction):
        if prediction is not None:
            disorder, color = self.DISORDER_MAPPING.get(prediction[0], ("Unknown", "gray"))
            st.markdown("<div class='result-text'>Prediction Result</div>",
                       unsafe_allow_html=True)
            st.markdown(f'<p style="color:{color};font-size:24px;text-align:center;'
                       f'font-weight:bold;">{disorder}</p>', unsafe_allow_html=True)
            
            self.display_recommendations(disorder)

    def display_recommendations(self, disorder):
        recommendations = {
            "Insomnia": [
                "Maintain a consistent sleep schedule",
                "Create a relaxing bedtime routine",
                "Avoid screens before bedtime",
                "Consider consulting a sleep specialist"
            ],
            "Sleep Apnea": [
                "Sleep on your side instead of your back",
                "Maintain a healthy weight",
                "Consider using a CPAP machine",
                "Consult a healthcare provider for proper diagnosis"
            ],
            "No Disorder": [
                "Continue maintaining good sleep habits",
                "Stay physically active",
                "Monitor sleep quality regularly",
                "Practice stress management"
            ]
        }
        
        if disorder in recommendations:
            st.markdown("### Recommendations:")
            for rec in recommendations[disorder]:
                st.markdown(f"- {rec}")

    def run(self):
        st.markdown('<p class="title">Sleep Disorder Detection</p>', unsafe_allow_html=True)
        
        submit_button, input_data = self.get_user_input()
        
        if submit_button:
            with st.spinner("Analyzing sleep patterns..."):
                input_df = pd.DataFrame(input_data)
                processed_data = self.preprocess_input(input_df)
                
                if processed_data is not None:
                    try:
                        prediction = self.model.predict(processed_data)
                        self.display_prediction(prediction)
                    except Exception as e:
                        st.error(f"Error making prediction: {e}")

if __name__ == "__main__":
    app = SleepDisorderApp()
    app.run()
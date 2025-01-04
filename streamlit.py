import streamlit as st
import pandas as pd
from src.pipeline.predict_pipeline import CustomData, PredictPipeline

# Set up Streamlit app
st.title("Student Performance Prediction")

# Home Page
st.sidebar.header("Navigation")
navigation = st.sidebar.radio("Go to", ["Home", "Predict"])

# Define the Home Page
if navigation == "Home":
    st.header("Welcome to the Student Performance Predictor")
    st.write(
        """
        This app predicts student performance based on various attributes like:
        - Gender
        - Ethnicity
        - Parental Level of Education
        - Lunch Type
        - Test Preparation Course
        - Writing and Reading Scores
        """
    )

# Define the Predict Page
if navigation == "Predict":
    st.header("Predict Student Performance")

    # Input Form
    with st.form("Prediction Form"):
        # Collect user input
        gender = st.selectbox("Gender", options=["male", "female"])
        ethnicity = st.selectbox("Race/Ethnicity", options=["group A", "group B", "group C", "group D", "group E"])
        parental_level_of_education = st.selectbox(
            "Parental Level of Education",
            options=[
                "some high school",
                "high school",
                "some college",
                "associate's degree",
                "bachelor's degree",
                "master's degree",
            ],
        )
        lunch = st.selectbox("Lunch Type", options=["standard", "free/reduced"])
        test_preparation_course = st.selectbox(
            "Test Preparation Course", options=["none", "completed"]
        )
        writing_score = st.number_input("Writing Score", min_value=0.0, max_value=100.0, step=1.0)
        reading_score = st.number_input("Reading Score", min_value=0.0, max_value=100.0, step=1.0)

        # Submit button
        submit = st.form_submit_button("Predict")

    if submit:
        # Create a `CustomData` object and get the data as a DataFrame
        data = CustomData(
            gender=gender,
            race_ethnicity=ethnicity,
            parental_level_of_education=parental_level_of_education,
            lunch=lunch,
            test_preparation_course=test_preparation_course,
            reading_score=reading_score,
            writing_score=writing_score,
        )

        pred_df = data.get_data_as_data_frame()
        st.write("Input Data:", pred_df)

        # Predict using the `PredictPipeline`
        predict_pipeline = PredictPipeline()
        results = predict_pipeline.predict(pred_df)

        # Display results
        st.success(f"Predicted Performance: {results[0]}")

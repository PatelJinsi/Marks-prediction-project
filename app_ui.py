import streamlit as st
import pickle
import pandas as pd

# Load the trained model
model = pickle.load(open('model.pkl', 'rb'))

st.title("Marks Prediction App")

# Input fields
study_hours = st.number_input("Enter Study Hours", min_value=0, max_value=24, value=1)
attendance = st.number_input("Enter Attendance Percentage", min_value=0, max_value=100, value=75)

# Predict button
if st.button("Predict"):
    # Use DataFrame for correct feature names
    input_data = pd.DataFrame([[study_hours, attendance]], columns=["study_hours", "attendance"])
    result = model.predict(input_data)
    
    # Access first element correctly
    st.write(f"Predicted Marks: {float(result[0]):.2f}")
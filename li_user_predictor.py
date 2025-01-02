import streamlit as st
import numpy as np
import pickle

# Load the trained logistic regression model
with open('li_model.pkl', 'rb') as file:
    li_model = pickle.load(file)

# App title and description
st.title("LinkedIn User Prediction App")
st.write("Enter the following details to predict whether you're likely to use LinkedIn and the probability of usage.")

# User inputs for model features
income = st.selectbox(
    "Income:",
    options=[1, 2, 3, 4, 5, 6, 7, 8, 9],
    format_func=lambda x: [
        "Less than $10,000",
        "$10,000 - $20,000",
        "$20,000 - $30,000",
        "$30,000 - $40,000",
        "$40,000 - $50,000",
        "$50,000 - $75,000",
        "$75,000 - $100,000",
        "$100,000 - $150,000",
        "More than $150,000"
    ][x - 1]
)
education = st.selectbox(
    "Education Level:",
    options=[1, 2, 3, 4, 5, 6, 7, 8],
    format_func=lambda x: [
        "Less than High School",
        "High School Graduate",
        "Some College",
        "Associate Degree",
        "Bachelor's Degree",
        "Master's Degree",
        "Professional Degree",
        "Doctorate"
    ][x - 1]
)
parent = st.radio("Are you a parent?", options=[0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
married = st.radio("Are you married?", options=[0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
female = st.radio("Are you female?", options=[0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
age = st.number_input("Age (1-98):", min_value=1, max_value=98, step=1)

# Prediction trigger
if st.button("Predict"):
    # Prepare feature array
    user_features = np.array([[income, education, parent, married, female, age]])

    # Make predictions
    predicted_class = li_model.predict(user_features)[0]
    predicted_prob = li_model.predict_proba(user_features)[0][1]

    # Display results
    st.write(f"Prediction: {'LinkedIn User' if predicted_class == 1 else 'Not a LinkedIn User'}")
    st.write(f"Probability of LinkedIn Usage: {predicted_prob:.2f}")

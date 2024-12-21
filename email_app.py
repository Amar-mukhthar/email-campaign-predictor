import streamlit as st
import pickle
import numpy as np
import pandas as pd

# Load the trained Random Forest model
with open('trained_rf_model.sav', 'rb') as model_file:
    loaded_model = pickle.load(model_file)

# Load the scaler
with open('scaler.pkl', 'rb') as scaler_file:
    scaler = pickle.load(scaler_file)

# Load the label encoder
with open('label_encoder.pkl', 'rb') as label_encoder_file:
    label_encoder = pickle.load(label_encoder_file)

# Define the function to process the age
def process_age(age):
    """
    Process raw age input to match the trained model's expectations.
    Args:
        age (int): Raw age input from the user.
    Returns:
        int: Encoded age group.
    """
    age_bins = [0, 18, 35, 50, 100]
    age_labels = ['teen', 'young_adult', 'middle_aged', 'senior']
    age_group = pd.cut([age], bins=age_bins, labels=age_labels)[0]
    age_group_encoded = label_encoder.transform([age_group])[0]
    return age_group_encoded

# Define the preprocessing function
def preprocess_input(age, emails_opened, emails_clicked, purchase_history, 
                     time_spent, days_since_last_open, engagement_score, 
                     device_type, clicked_previous_emails):
    """
    Preprocess raw inputs for prediction.
    Args:
        Various inputs for customer attributes.
    Returns:
        np.array: Preprocessed input ready for prediction.
    """
    age_group_encoded = process_age(age)
    numerical_data = scaler.transform([[emails_opened, emails_clicked, purchase_history,
                                        time_spent, days_since_last_open, engagement_score]])
    final_input = np.hstack([
        [device_type],             # Device type (binary)
        [clicked_previous_emails], # Clicked Previous Emails (binary)
        [age_group_encoded],       # Encoded age group
        numerical_data[0]          # Standardized numerical features
    ])
    return final_input

# Streamlit App Interface
st.title("Email Campaign Success Predictor")

st.header("Enter Customer Details:")
age = st.number_input("Customer Age:", min_value=0, step=1)
emails_opened = st.number_input("Number of Emails Opened:", min_value=0, step=1)
emails_clicked = st.number_input("Number of Emails Clicked:", min_value=0, step=1)
purchase_history = st.number_input("Purchase History ($):", min_value=0.0)
time_spent = st.number_input("Time Spent on Website (minutes):", min_value=0.0)
days_since_last_open = st.number_input("Days Since Last Email Opened:", min_value=0)
engagement_score = st.number_input("Engagement Score:", min_value=0.0)
device_type = st.radio("Device Type:", options=[0, 1], format_func=lambda x: "Mobile" if x == 1 else "Desktop")
clicked_previous_emails = st.radio("Clicked Previous Emails?", options=[0, 1], format_func=lambda x: "Yes" if x == 1 else "No")

# Prediction Button
if st.button("Predict"):
    # Preprocess the input
    processed_data = preprocess_input(
        age, emails_opened, emails_clicked, purchase_history,
        time_spent, days_since_last_open, engagement_score, 
        device_type, clicked_previous_emails
    )

    # Predict using the trained model
    prediction = loaded_model.predict(processed_data.reshape(1, -1))

    # Display the result
    if prediction[0] == 1:
        st.success("The customer is likely to open the email!")
    else:
        st.error("The customer is unlikely to open the email.")

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
    
    numerical_data = scaler.transform([[age,emails_opened, emails_clicked, purchase_history,
                                        time_spent, days_since_last_open, engagement_score]])
    final_input = np.hstack([
        [device_type],             # Device type (binary)
        [clicked_previous_emails], # Clicked Previous Emails (binary)
        numerical_data[0]          # Standardized numerical features
    ])
    return final_input

# Streamlit App Interface
st.title("Email Campaign Success Predictor")
# Add an introductory explanation

# Add an introductory explanation
st.write("""
This app helps predict if a customer will open an email in a marketing campaign. You enter customer details like age, past email activity, purchase history, website time, and device type. The app uses this information to guess if they'll open the next email. It's a userfriendly way to predict email campaign success and helps marketing teams make better decisions.
""")


st.header("Enter Customer Details:")
age = st.number_input("Customer Age:", min_value=0, step=1)
emails_opened = st.number_input("Number of Emails Opened:", min_value=0, step=1,help="How many emails the customer has opened previously.")
emails_clicked = st.number_input("Number of Emails Clicked:", min_value=0, step=1,help="How many emails the customer has clicked on after opening them.")
purchase_history = st.number_input("Purchase History ($):", min_value=0.0, help="Total amount of money the customer has spent in past purchases.")
time_spent = st.number_input("Time Spent on Website (minutes):", min_value=0.0, help="Average time the customer spends on the website after receiving marketing emails.")
days_since_last_open = st.number_input("Days Since Last Email Opened:", min_value=0,help="Number of days since the customer last opened an email.")
engagement_score = st.number_input("Engagement Score:", min_value=0.0,help="A score reflecting overall customer engagement, calculated based on interactions with emails and website activity.")
device_type = st.radio("Device Type:", options=[0, 1], format_func=lambda x: "Mobile" if x == 1 else "Desktop")
clicked_previous_emails = st.radio("Clicked Previous Emails?", options=[0, 1], format_func=lambda x: "Yes" if x == 1 else "No",help="Whether the customer has clicked on links in previous emails")

# Updated Prediction Button Logic
if st.button("Predict"):
    try:
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

    except ValueError as e:
        st.error(f"Input Error: {e}")

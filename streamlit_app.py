import altair as alt
import pandas as pd
import streamlit as st
import joblib
import numpy as np
import mlflow

# Set up MLflow tracking
mlflow.set_tracking_uri("http://127.0.0.1:5000")  # Change this to your MLflow server URI if needed
mlflow.set_experiment("House_Price_Prediction")

# Show the page title and description.
st.set_page_config(page_title="House Price Prediction", page_icon="🏡")
st.title("🏡 House Price Prediction")
st.write(
    """
    This app predicts the price of a house based on its features such as size, 
    number of bedrooms, and bathrooms using a linear regression model. Just 
    click on the widgets below to explore!
    """
)

# Load the model
# Load the data from a CSV. We're caching this so it doesn't reload every time the app
# reruns (e.g. if the user interacts with the widgets).
@st.cache_resource
def load_model():
    model = joblib.load("house_price_model.pkl")
    return model


model = load_model()

# User inputs for house features
st.write("Enter the details of the house:")

size = st.number_input("Size (square feet)", min_value=500, max_value=10000, step=100)
bedrooms = st.number_input("Number of Bedrooms", min_value=1, max_value=10, step=1)
bathrooms = st.number_input("Number of Bathrooms", min_value=1, max_value=10, step=1)

# Predict button
if st.button("Predict Price"):
    # End any active MLflow run before starting a new one
    mlflow.end_run()
    
    # Start a new MLflow run
    with mlflow.start_run():
        mlflow.log_param("size", size)
        mlflow.log_param("bedrooms", bedrooms)
        mlflow.log_param("bathrooms", bathrooms)

    # Prepare the input features as a 2D array
    input_features = np.array([[size, bedrooms, bathrooms]])
    
    # Make the prediction using the model
    predicted_price = model.predict(input_features)[0]
   
    # Log the prediction
    mlflow.log_metric("predicted_price", predicted_price)
    
    # Display the predicted price
    st.write(f"The predicted price of the house is **${predicted_price:,.2f}**")



# Additional feature: Display information about the model's performance
st.write("---")
st.write("### Model Information")
st.write(
    """
    The linear regression model was trained using a dataset of house prices and 
    various features. It predicts the price based on the size, number of bedrooms, 
    and number of bathrooms. 
    """
)
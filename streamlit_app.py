import subprocess
import time

import requests
import streamlit as st

# Define the API endpoint URL
API_URL = "http://localhost:8132/predict"


# Define a function to start the FastAPI server using subprocess
def start_fastapi_server():
    subprocess.Popen(["uvicorn", "app:app", "--reload"])


# Define a function to check if the FastAPI server is running
def is_fastapi_running():
    try:
        response = requests.get(API_URL)
        return response.ok
    except:
        return False


# Check if the FastAPI server is already running
if not is_fastapi_running():
    # If not, start the server in a separate process
    st.write("Starting FastAPI server...")
    start_fastapi_server()
    # Wait for the server to start up
    time.sleep(3)


# Define the input field for the user's text input
input_text = st.text_input("Enter a news headline or article text")


# Define a button to submit the text for prediction
predict_button = st.button("Predict")


# When the user clicks the "Predict" button...
if predict_button:
    # Show a message while waiting for the prediction result
    with st.spinner("Making prediction..."):
        # Send the input text to the API endpoint
        response = requests.post(API_URL, json={"text": input_text})
    # If the response was successful, show the prediction result
    if response.ok:
        # Get the prediction result from the response
        prediction = response.json()["prediction"]
        # Show the prediction result
        st.write(f"Prediction: {prediction}")

    # Otherwise, show an error message
    else:
        st.write("Error making prediction. Please try again later.")


# Add some custom CSS to style the app
st.markdown("""
<style>
.sidebar .sidebar-content {
    background-color: #f8f9fa;
}
.btn-primary {
    background-color: #0056b3;
    border-color: #0056b3;
}
.btn-primary:hover {
    background-color: #003a7d;
    border-color: #003a7d;
}
</style>
""", unsafe_allow_html=True)

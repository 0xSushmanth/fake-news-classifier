import pickle

import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
import numpy as np
import tensorflow as tf
from starlette import status
from starlette.responses import JSONResponse
from tensorflow.keras.layers import Hashing

import subprocess
import time
import requests
import streamlit as st

# Create a FastAPI app instance
app = FastAPI()


# Define the input schema
class InputText(BaseModel):
    text: str


# Create a tokenizer
vocab_size = 10000
tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=vocab_size)

# Load the pre-trained model
with open("model.pkl", "rb") as f:
    model = pickle.load(f)


# Define a model to perform text preprocessing
class TextPreprocessor:

    def __init__(self, input_length=20):
        self.input_length = input_length

    # Fit the tokenizer on the input text
    def fit(self, input_text: List[str]):
        tokenizer.fit_on_texts(input_text)

    # Convert input text to sequences
    def transform(self, input_text: List[str]) -> np.ndarray:
        sequences = tokenizer.texts_to_sequences(input_text)
        # Pad sequences to a fixed length
        padded_sequences = tf.keras.preprocessing.sequence.pad_sequences(
            sequences, maxlen=self.input_length, padding="post"
        )
        # Hash the padded sequences into one-hot vectors
        hashing_layer = Hashing(vocab_size)
        one_hot_vectors = hashing_layer(padded_sequences).numpy()
        return one_hot_vectors


# Create an instance of the text preprocessor
preprocessor = TextPreprocessor()


@app.post("/predict")
async def predict(input_text: InputText):
    # Fit the tokenizer on the input text
    preprocessor.fit([input_text.text])
    # Transform the input text into one-hot encoded vectors
    one_hot_vectors = preprocessor.transform([input_text.text])
    # Make a prediction using the pre-trained model
    raw_prediction = model.predict(one_hot_vectors)
    # Convert the raw prediction to a human-readable label
    if raw_prediction[0] == 1:
        prediction_label = "Real News"
    else:
        prediction_label = "Fake News"
    return {"prediction": prediction_label}


class ErrorResponse(BaseModel):
    detail: List[BaseModel]


@app.exception_handler(Exception)
async def exception_handler(request, exc):
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"detail": [ErrorResponse(detail=str(exc))]},
    )


# Define the API endpoint URL
API_URL = "http://localhost:8000/predict"


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

# Create a FastAPI app instance
app = FastAPI()

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)

import pickle
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import tensorflow as tf
from starlette import status
from starlette.responses import JSONResponse
from tensorflow.keras.layers import Hashing

app = FastAPI()

# Define the input schema
class InputText(BaseModel):
    text: str

# Load the pre-trained model
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

# Create a tokenizer
vocab_size = 10000
tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=vocab_size)

# Define a model to perform text preprocessing
class TextPreprocessor:
    def __init__(self, input_length=20):
        self.input_length = input_length

    # Fit the tokenizer on the input text
    def fit(self, input_text: list):
        tokenizer.fit_on_texts(input_text)

    # Transform the input text into one-hot encoded vectors
    def transform(self, input_text: list) -> np.ndarray:
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

@app.get('/')
def index():
    return "Try to use /predict endpoint"

@app.post("/predict")
async def predict(input_text: InputText):
    # Transform the input text into one-hot encoded vectors
    one_hot_vectors = preprocessor.transform([input_text.text])
    # Make a prediction using the pre-trained model
    raw_prediction = model.predict(one_hot_vectors)
    # Convert the raw prediction to a human-readable label
    prediction_label = "Real News" if raw_prediction[0] == 1 else "Fake News"
    return {"prediction": prediction_label}

@app.exception_handler(Exception)
async def exception_handler(request, exc):
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"detail": [{"detail": str(exc)}]},
    )

if __name__ == '__main__':
    uvicorn.run(app, host='0.0.0.0', port=8000)

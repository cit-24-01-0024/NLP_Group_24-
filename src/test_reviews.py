import numpy as np
import joblib

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences


# ==========================
# Load saved Bi-LSTM model
# ==========================

model = load_model(
    "models/bilstm.keras"
)


# Load tokenizer
tokenizer = joblib.load(
    "models/tokenizer.pkl"
)


# Sentiment labels
labels = [
    "Negative",
    "Neutral",
    "Positive"
]


# ==========================
# Function for prediction
# ==========================

def predict_sentiment(review):

    # Convert review into sequence
    sequence = tokenizer.texts_to_sequences([review])

    # Padding
    padded = pad_sequences(
        sequence,
        maxlen=100
    )

    # Prediction
    prediction = model.predict(padded)

    result = np.argmax(prediction)

    return labels[result]



# ==========================
# Test new reviews
# ==========================

reviews = [
    "I absolutely love this phone.",
    "Worst product ever.",
    "The quality is okay."
]


for review in reviews:

    sentiment = predict_sentiment(review)

    print("\nReview:")
    print(review)

    print("Prediction:")
    print(sentiment)
import os
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Embedding,
    Bidirectional,
    LSTM,
    Dense,
    Dropout
)

from tensorflow.keras.utils import to_categorical

# Load processed dataset

df = pd.read_csv(
    "data/processed_reviews.csv"
)

os.makedirs("results", exist_ok=True)

X = df["clean_review"]

y = df["sentiment"]

label_encoder = LabelEncoder()

y = label_encoder.fit_transform(y)


print(label_encoder.classes_)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


print("Training samples:", len(X_train))
print("Testing samples:", len(X_test))

tokenizer = Tokenizer(
    num_words=5000
)


tokenizer.fit_on_texts(X_train)


X_train_seq = tokenizer.texts_to_sequences(X_train)

X_test_seq = tokenizer.texts_to_sequences(X_test)

max_length = 100


X_train_pad = pad_sequences(
    X_train_seq,
    maxlen=max_length
)


X_test_pad = pad_sequences(
    X_test_seq,
    maxlen=max_length
)

y_train_cat = to_categorical(
    y_train,
    num_classes=3
)


y_test_cat = to_categorical(
    y_test,
    num_classes=3
)

model = Sequential()


model.add(
    Embedding(
        input_dim=5000,
        output_dim=128,
        input_length=max_length
    )
)


model.add(
    Bidirectional(
        LSTM(64)
    )
)


model.add(
    Dropout(0.5)
)


model.add(
    Dense(
        3,
        activation="softmax"
    )
)


model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)


model.summary()

history = model.fit(
    X_train_pad,
    y_train_cat,
    epochs=10,
    batch_size=32,
    validation_split=0.2
)

y_pred_prob = model.predict(
    X_test_pad
)


y_pred = np.argmax(
    y_pred_prob,
    axis=1
)

print("\n===== Bi-LSTM Evaluation =====")


print(
    "Accuracy:",
    accuracy_score(y_test, y_pred)
)


print(
    "Precision:",
    precision_score(
        y_test,
        y_pred,
        average="weighted",
        zero_division=0
    )
)


print(
    "Recall:",
    recall_score(
        y_test,
        y_pred,
        average="weighted",
        zero_division=0
    )
)


print(
    "F1 Score:",
    f1_score(
        y_test,
        y_pred,
        average="weighted",
        zero_division=0
    )
)


print("\nClassification Report:")

report = classification_report(
    y_test,
    y_pred,
    zero_division=0
)

print(report)

with open("results/bilstm_report.txt", "w") as file:
    file.write(report)


print("\nConfusion Matrix:")

cm = confusion_matrix(
    y_test,
    y_pred
)

print(cm)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=label_encoder.classes_
)

disp.plot()

plt.title("Bi-LSTM Confusion Matrix")

plt.savefig("results/bilstm_confusion_matrix.png")

plt.close()

model.save(
    "models/bilstm.keras"
)

joblib.dump(
    tokenizer,
    "models/tokenizer.pkl"
)

print("Bi-LSTM model saved successfully")

plt.figure(figsize=(6,4))

plt.plot(
    history.history["accuracy"],
    label="Training Accuracy"
)

plt.plot(
    history.history["val_accuracy"],
    label="Validation Accuracy"
)

plt.title("Bi-LSTM Accuracy")

plt.xlabel("Epoch")

plt.ylabel("Accuracy")

plt.legend()

plt.savefig("results/bilstm_accuracy.png")

plt.close()

print("Tokenizer saved successfully")
print("Accuracy graph saved successfully")
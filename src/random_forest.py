import pandas as pd
from sklearn.model_selection import train_test_split


# Load processed dataset
df = pd.read_csv("data/processed_reviews.csv")


# Separate input and output

# X = Review text
X = df["clean_review"]

# y = Sentiment labels
y = df["sentiment"]


# Split dataset into 80% training and 20% testing

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


# Display split sizes

print("Total samples:", len(df))

print("Training samples:", len(X_train))

print("Testing samples:", len(X_test))


print("\nTraining data example:")
print(X_train.head())

print("\nTraining labels example:")
print(y_train.head())

import os
import joblib
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)


# ==============================
# Load processed dataset
# ==============================

df = pd.read_csv("data/processed_reviews.csv")

os.makedirs("results", exist_ok=True)

# ==============================
# Separate input and output
# ==============================

# X = Review text
X = df["clean_review"]

# y = Sentiment labels
y = df["sentiment"]


# ==============================
# Split dataset into 80% training and 20% testing
# ==============================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


print("Total samples:", len(df))
print("Training samples:", len(X_train))
print("Testing samples:", len(X_test))


# ==============================
# TF-IDF Conversion
# ==============================

tfidf = TfidfVectorizer(
    max_features=5000
)


# Convert text into numerical values

X_train_tfidf = tfidf.fit_transform(X_train)

X_test_tfidf = tfidf.transform(X_test)


print("\nTF-IDF conversion completed")
print("Training shape:", X_train_tfidf.shape)
print("Testing shape:", X_test_tfidf.shape)



# ==============================
# Random Forest Model
# ==============================

rf_model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)


# Train model

rf_model.fit(
    X_train_tfidf,
    y_train
)


print("\nRandom Forest training completed")



# ==============================
# Prediction
# ==============================

y_pred = rf_model.predict(X_test_tfidf)



# ==============================
# Model Evaluation
# ==============================

print("\n===== Model Evaluation =====")


accuracy = accuracy_score(
    y_test,
    y_pred
)

precision = precision_score(
    y_test,
    y_pred,
    average="weighted",
    zero_division=0
)

recall = recall_score(
    y_test,
    y_pred,
    average="weighted",
    zero_division=0
)

f1 = f1_score(
    y_test,
    y_pred,
    average="weighted",
    zero_division=0
)


print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)



print("\nClassification Report:")

report = classification_report(
    y_test,
    y_pred,
    zero_division=0
)

print(report)

with open("results/random_forest_report.txt", "w") as file:
    file.write(report)



print("\nConfusion Matrix:")

cm = confusion_matrix(
    y_test,
    y_pred
)

print(cm)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=rf_model.classes_
)

disp.plot()

plt.title("Random Forest Confusion Matrix")

plt.savefig("results/random_forest_confusion_matrix.png")

plt.close()



# ==============================
# Save Model
# ==============================

joblib.dump(
    rf_model,
    "models/random_forest.pkl"
)


joblib.dump(
    tfidf,
    "models/tfidf_vectorizer.pkl"
)


print("\nModel saved successfully")
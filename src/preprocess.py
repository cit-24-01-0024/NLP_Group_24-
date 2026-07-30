import pandas as pd

# Load the dataset
df = pd.read_csv("data/amazon.csv")

# Show the first 5 rows
print(df.head())

# Show the dataset shape
print(df.shape)

# Show the column names
print(df.columns.tolist())

# Show review title
print("\n===== Review Title =====")
print(df["review_title"].head())

# Show review content
print("\n===== Review Content =====")
print(df["review_content"].head())

# Show ratings
print("\n===== Rating =====")
print(df["rating"].head())

# Convert the rating column from text to numbers
df["rating"] = pd.to_numeric(df["rating"], errors="coerce")

# Function to convert rating into sentiment
def get_sentiment(rating):
    if rating < 3:
        return "Negative"
    elif rating < 4:
        return "Neutral"
    else:
        return "Positive"

# Create a new sentiment column
df["sentiment"] = df["rating"].apply(get_sentiment)

# Display the new column
print("\n===== Sentiment Labels =====")
print(df[["rating", "sentiment"]].head(10))

print("\n===== Unique Ratings =====")
print(sorted(df["rating"].dropna().unique())) 

import pandas as pd
import re
import nltk

from nltk.corpus import stopwords

nltk.download('stopwords')

# ==============================
# Data Cleaning
# ==============================

# Combine review title and review content
df["review_text"] = (
    df["review_title"].fillna("") + " " +
    df["review_content"].fillna("")
)


# Remove empty reviews
df = df[df["review_text"].str.strip() != ""]


# Remove duplicate reviews
df = df.drop_duplicates(subset=["review_text"])


# Stop words list
stop_words = set(stopwords.words("english"))


# Text cleaning function
def clean_text(text):

    # Convert to lowercase
    text = text.lower()

    # Remove URLs
    text = re.sub(r"http\S+|www\S+", "", text)

    # Remove punctuation, numbers, special characters
    text = re.sub(r"[^a-zA-Z\s]", " ", text) 

    # Remove extra spaces
    text = re.sub(r"\s+", " ", text).strip()

    # Remove stop words
    words = text.split()

    words = [
        word for word in words 
        if word not in stop_words
    ]

    return " ".join(words)


# Apply cleaning
df["clean_review"] = df["review_text"].apply(clean_text)


# Show before and after cleaning
print("\n===== Before Cleaning =====")
print(df["review_text"].head())

print("\n===== After Cleaning =====")
print(df["clean_review"].head())


# Save processed dataset
df[["clean_review", "sentiment"]].to_csv(
    "data/processed_reviews.csv",
    index=False
)

print("\nProcessed dataset saved successfully")
print(df[["clean_review", "sentiment"]].shape)

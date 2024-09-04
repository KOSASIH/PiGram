import pandas as pd
import numpy as np
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Load the data from a CSV file
def load_data(file_path):
    data = pd.read_csv(file_path)
    return data

# Preprocess the text data
def preprocess_text(data):
    nltk.download("vader_lexicon")
    sia = SentimentIntensityAnalyzer()
    data["sentiment"] = data["text"].apply(lambda x: sia.polarity_scores(x)["compound"])
    return data

# Train a Naive Bayes classifier for sentiment analysis
def train_sentiment_analysis_model(data):
    X = data["text"]
    y = data["sentiment"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    clf = MultinomialNB()
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Classification Report:")
    print(classification_report(y_test, y_pred))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    return clf

# Example usage
if __name__ == "__main__":
    data = load_data("data.csv")
    data = preprocess_text(data)
    clf = train_sentiment_analysis_model(data)

"""
predict.py — Deployment Demo (Bonus)

A minimal command-line demonstration of the trained sentiment model in action.
It loads the saved model + TF-IDF vectorizer from disk (the same artifacts produced
by the notebook) and classifies new, unseen customer reviews.

Usage:
    python predict.py "This dress is gorgeous and fits perfectly!"
    python predict.py                      # runs a few built-in example reviews instead

Requires: sentiment_model.pkl and tfidf_vectorizer.pkl in the same folder,
and the same cleaning function used during training (re-implemented below so
this script has no dependency on the notebook itself).
"""

import sys
import re
import pickle

import nltk
from nltk.corpus import stopwords

try:
    stop_words = set(stopwords.words('english'))
except LookupError:
    nltk.download('stopwords')
    stop_words = set(stopwords.words('english'))


def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-z\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    tokens = text.split()
    tokens = [t for t in tokens if t not in stop_words and len(t) > 2]
    return ' '.join(tokens)


def load_artifacts(model_path='sentiment_model.pkl', vectorizer_path='tfidf_vectorizer.pkl'):
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    with open(vectorizer_path, 'rb') as f:
        vectorizer = pickle.load(f)
    return model, vectorizer


def predict_sentiment(review_text, model, vectorizer):
    cleaned = clean_text(review_text)
    vector = vectorizer.transform([cleaned])
    prediction = model.predict(vector)[0]
    if hasattr(model, "predict_proba"):
        probabilities = dict(zip(model.classes_, model.predict_proba(vector)[0].round(3)))
    else:
        probabilities = None
    return prediction, probabilities


DEMO_REVIEWS = [
    "This dress is absolutely gorgeous, fits perfectly and the fabric feels premium!",
    "It's okay, the color is nice but the fit runs a bit small for the price.",
    "Terrible quality, the seams fell apart after one wash. Very disappointed.",
    "Cute top but definitely runs large, ordered a size down and it was perfect.",
]


def main():
    model, vectorizer = load_artifacts()

    if len(sys.argv) > 1:
        reviews = [" ".join(sys.argv[1:])]
    else:
        print("No review text provided as an argument — running built-in demo reviews instead.\n")
        reviews = DEMO_REVIEWS

    for review in reviews:
        label, probs = predict_sentiment(review, model, vectorizer)
        print(f"Review: {review}")
        print(f"  -> Predicted sentiment: {label}")
        if probs:
            print(f"  -> Probabilities: {probs}")
        print()


if __name__ == "__main__":
    main()

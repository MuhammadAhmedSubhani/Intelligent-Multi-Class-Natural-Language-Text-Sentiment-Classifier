import joblib
from pathlib import Path
from preprocessing import preprocess_text


def main():
    BASE_DIR = Path(__file__).resolve().parent.parent
    MODEL_DIR = BASE_DIR / "models"

    # Load trained model and TF-IDF vectorizer
    model = joblib.load(MODEL_DIR / "sentiment_model.pkl")
    vectorizer = joblib.load(MODEL_DIR / "tfidf_vectorizer.pkl")

    print("Model and vectorizer loaded successfully.")

    # Get text from user
    text = input("\nEnter a sentence: ")

    # Preprocess the text
    cleaned_text = preprocess_text(text)

    # Convert text into TF-IDF numbers
    text_vector = vectorizer.transform([cleaned_text])

    # Predict sentiment
    prediction = model.predict(text_vector)

    # Get prediction probabilities
    probabilities = model.predict_proba(text_vector)[0]

    print("\nPredicted Sentiment:", prediction[0])

    print("\nPrediction Probabilities:")

    for sentiment, probability in zip(model.classes_, probabilities):
        print(f"{sentiment}: {probability * 100:.2f}%")


if __name__ == "__main__":
    main()
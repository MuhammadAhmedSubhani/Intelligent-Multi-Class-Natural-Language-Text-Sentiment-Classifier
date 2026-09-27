import pandas as pd
import joblib

from preprocessing import preprocess_text
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from pathlib import Path


def main():

    # Project root folder
    BASE_DIR = Path(__file__).resolve().parent.parent

    # Models folder
    MODEL_DIR = BASE_DIR / "models"

    # Load the local training dataset
    df = pd.read_csv(
        BASE_DIR / "data" / "dataset.csv"
    )

    print(f"Dataset loaded successfully: {df.shape}")

    # Preprocess the text
    df["cleaned_text"] = df["text"].apply(preprocess_text)

    # Define input features and target labels
    X = df["cleaned_text"]
    y = df["sentiment"]

    # Split data into training and testing sets
    X_train_text, X_test_text, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    print(f"Training samples: {len(X_train_text)}")
    print(f"Testing samples: {len(X_test_text)}")

    # Create TF-IDF vectorizer
    vectorizer = TfidfVectorizer()

    # Learn TF-IDF only from training text
    X_train = vectorizer.fit_transform(X_train_text)

    # Transform test text using the same TF-IDF vocabulary
    X_test = vectorizer.transform(X_test_text)

    print(f"TF-IDF training shape: {X_train.shape}")
    print(f"TF-IDF testing shape: {X_test.shape}")

    # Ensure the models directory exists
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    # Save test data for evaluation
    test_data_file_path = MODEL_DIR / "test_data.pkl"

    joblib.dump(
        {
            "X_test": X_test,
            "y_test": y_test
        },
        test_data_file_path
    )

    print(f"Test data saved successfully to: {test_data_file_path}")

    # Create Logistic Regression classifier
    model = LogisticRegression(
        max_iter=1000
    )

    # Train the model
    model.fit(X_train, y_train)

    print("Model training completed.")

    # Save the trained model
    model_file_path = MODEL_DIR / "sentiment_model.pkl"

    joblib.dump(model, model_file_path)

    print(f"Model saved successfully to: {model_file_path}")

    # Save the vectorizer separately
    vectorizer_file_path = MODEL_DIR / "tfidf_vectorizer.pkl"

    joblib.dump(vectorizer, vectorizer_file_path)

    print(f"Vectorizer saved successfully to: {vectorizer_file_path}")


if __name__ == "__main__":
    main()
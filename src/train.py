import pandas as pd
import joblib

from preprocessing import preprocess_text
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from pathlib import Path

def main():

    # Load the training dataset
    df = pd.read_csv(
        "hf://datasets/Sp1786/multiclass-sentiment-analysis-dataset/train_df.csv"
    )

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

    # Create TF-IDF vectorizer
    vectorizer = TfidfVectorizer()

    # Learn TF-IDF only from training text
    X_train = vectorizer.fit_transform(X_train_text)

    # Transform test text using the same TF-IDF vocabulary
    X_test = vectorizer.transform(X_test_text)
    # Path(__file__) is '.../src/train.py'
    # .parent is '.../src'
    # .parent.parent is your project root folder
    BASE_DIR = Path(__file__).resolve().parent.parent
    MODEL_DIR = BASE_DIR / "models"

    # Ensure the models directory exists automatically
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
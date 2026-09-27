import pandas as pd
import joblib

from preprocessing import preprocess_text
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

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

    # Create Logistic Regression classifier
    model = LogisticRegression(
        max_iter=1000
    )

    # Train the model
    model.fit(X_train, y_train)
    print("Model training completed.")
    # Save the trained model and TF-IDF vectorizer
    joblib.dump(
        {
            "model": model,
            "vectorizer": vectorizer
        },
        "../models/sentiment_model.pkl"
    )

    print("Model and vectorizer saved successfully.")

    # Make predictions on the test data
    predictions = model.predict(X_test)
    
    # Calculate accuracy
    accuracy = accuracy_score(y_test, predictions)
    print("Accuracy:", accuracy)

    # Generate classification report
    report = classification_report(y_test, predictions)
    print("\nClassification Report:")
    print(report)



if __name__ == "__main__":
    main()
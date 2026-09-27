import joblib
import matplotlib.pyplot as plt
import seaborn as sns

from pathlib import Path
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score
)

def main():

    # Find the project root directory
    BASE_DIR = Path(__file__).resolve().parent.parent

    # Path to the models folder
    MODEL_DIR = BASE_DIR / "models"

    # Load the trained model
    model = joblib.load(
        MODEL_DIR / "sentiment_model.pkl"
    )

    # Load the test data
    test_data = joblib.load(
        MODEL_DIR / "test_data.pkl"
    )

    X_test = test_data["X_test"]
    y_test = test_data["y_test"]

    print("Model and test data loaded successfully.")

    # Make predictions
    predictions = model.predict(X_test)

    print("Predictions completed.")

    # Calculate accuracy
    accuracy = accuracy_score(
        y_test,
        predictions
    )

    print("\nAccuracy:", accuracy)

    # Generate classification report
    report = classification_report(
        y_test,
        predictions
    )

    print("\nClassification Report:")
    print(report)

    # Calculate F1-scores
    macro_f1 = f1_score(
        y_test,
        predictions,
        average="macro"
    )

    weighted_f1 = f1_score(
        y_test,
        predictions,
        average="weighted"
    )

    # Create confusion matrix
    labels = ["negative", "neutral", "positive"]

    cm = confusion_matrix(
        y_test,
        predictions,
        labels=labels
    )

    print("\nConfusion Matrix:")
    print(cm)

    # Create results directory
    RESULTS_DIR = BASE_DIR / "results"
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # Save classification report
    report_path = RESULTS_DIR / "classification_report.txt"

    with open(report_path, "w") as file:
        file.write(report)

    print(f"Classification report saved to: {report_path}")


    # Save main evaluation metrics
    metrics_path = RESULTS_DIR / "metrics.txt"

    with open(metrics_path, "w") as file:
        file.write(f"Accuracy: {accuracy:.4f}\n")
        file.write(f"Macro F1-score: {macro_f1:.4f}\n")
        file.write(f"Weighted F1-score: {weighted_f1:.4f}\n")

    print(f"Metrics saved to: {metrics_path}")
    
    # Plot confusion matrix
    plt.figure(figsize=(7, 5))

    sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=labels,
    yticklabels=labels
    )

    plt.xlabel("Predicted Label")
    plt.ylabel("Actual Label")
    plt.title("Sentiment Classification Confusion Matrix")

    # Save confusion matrix
    cm_path = RESULTS_DIR / "confusion_matrix.png"
    plt.savefig(cm_path, bbox_inches="tight")

    plt.close()

    print(f"Confusion matrix saved to: {cm_path}")


if __name__ == "__main__":
    main()
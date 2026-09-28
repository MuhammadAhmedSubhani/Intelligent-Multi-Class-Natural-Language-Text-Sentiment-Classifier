from flask import Flask, request, jsonify, send_from_directory
import joblib
from pathlib import Path
import sys

BASE_DIR = Path(__file__).resolve().parent
SRC_DIR = BASE_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from preprocessing import preprocess_text

app = Flask(__name__, static_folder="frontend", static_url_path="")
MODEL_DIR = BASE_DIR / "models"
MODEL_PATH = MODEL_DIR / "sentiment_model.pkl"
VECTORIZER_PATH = MODEL_DIR / "tfidf_vectorizer.pkl"

model = None
vectorizer = None

def load_artifacts():
    global model, vectorizer
    model = joblib.load(MODEL_PATH)
    vectorizer = joblib.load(VECTORIZER_PATH)

@app.route("/")
def home():
    return send_from_directory(app.static_folder, "index.html")

@app.route("/api/health")
def health():
    return jsonify({
        "status": "ok",
        "model_loaded": model is not None,
        "vectorizer_loaded": vectorizer is not None
    })

@app.route("/api/predict", methods=["POST"])
def predict():
    data = request.get_json(silent=True) or {}
    text = str(data.get("text", "")).strip()

    if not text:
        return jsonify({"error": "Please enter some text."}), 400
    if len(text) > 2000:
        return jsonify({"error": "Text must be 2000 characters or fewer."}), 400

    cleaned_text = preprocess_text(text)
    text_vector = vectorizer.transform([cleaned_text])
    prediction = model.predict(text_vector)[0]
    probabilities = model.predict_proba(text_vector)[0]

    return jsonify({
        "sentiment": str(prediction),
        "probabilities": {
            sentiment: round(float(probability) * 100, 2)
            for sentiment, probability in zip(model.classes_, probabilities)
        }
    })

if __name__ == "__main__":
    load_artifacts()
    print("Model and TF-IDF vectorizer loaded successfully.")
    print("Open http://127.0.0.1:5000")
    app.run(debug=True)

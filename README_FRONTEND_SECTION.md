## 🖥️ Interactive Web Frontend

The project includes a browser interface connected directly to the trained Scikit-Learn sentiment model.

### Features
- Interactive text input and character counter
- Negative / Neutral / Positive prediction
- Probability display for all three classes
- Example sentences
- Responsive modern dashboard
- Visual ML pipeline
- Flask API using the actual saved model and TF-IDF vectorizer

### Architecture

```text
Browser → HTML/CSS/JavaScript → Flask API
       → NLTK Preprocessing → TF-IDF → Logistic Regression
       → Sentiment + Probabilities
```

### Run

```bash
pip install flask
python app.py
```

Then open `http://127.0.0.1:5000`.

The frontend uses `models/sentiment_model.pkl` and `models/tfidf_vectorizer.pkl`; predictions are not hard-coded.

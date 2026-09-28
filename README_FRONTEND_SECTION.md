# Frontend / Flask Web Application

This document describes the additional web interface built for the **Intelligent Multi-Class Natural Language Text Sentiment Classifier**.

## Overview

The frontend turns the trained NLP model into an interactive browser-based sentiment analysis application.

It uses:

- **Flask** for the Python backend
- **HTML** for page structure
- **CSS** for styling and responsive layout
- **JavaScript** for interaction and API communication
- The project's existing **NLTK preprocessing pipeline**
- The project's existing **TF-IDF vectorizer**
- The project's existing **Logistic Regression model**

The frontend does **not** use hard-coded sentiment results. It calls the actual trained model stored in the `models/` directory.

---

## Architecture

```text
Browser
   │
   │ POST /api/predict
   ↓
Flask Backend
   │
   ├── preprocess_text()
   │
   ├── TF-IDF vectorizer
   │
   └── Logistic Regression model
   │
   ↓
Prediction + Probabilities
   │
   │ JSON response
   ↓
JavaScript
   │
   ↓
Dashboard Result
```

---

## Main Files

```text
frontend/
├── index.html
├── script.js
└── style.css

app.py
```

### `app.py`

The Flask application:

1. Loads the saved Logistic Regression model.
2. Loads the saved TF-IDF vectorizer.
3. Imports the project's preprocessing function.
4. Serves the frontend.
5. Provides the prediction API.
6. Returns sentiment probabilities.

### `frontend/index.html`

Contains the dashboard interface:

- Text input
- Character counter
- Example buttons
- Analyze button
- Prediction result
- Probability bars
- Model information
- Project metrics
- Pipeline visualization

### `frontend/style.css`

Controls:

- Dark dashboard theme
- Cards
- Typography
- Buttons
- Probability bars
- Spacing
- Responsive layout

### `frontend/script.js`

Handles:

- Reading the user's text
- Character counting
- Example text buttons
- Sending API requests
- Receiving JSON results
- Updating the prediction display
- Updating probability bars

---

## API

### Health Check

```text
GET /api/health
```

Used to check whether the Flask application is running.

### Prediction

```text
POST /api/predict
```

Example request:

```json
{
    "text": "I absolutely love this product!"
}
```

The backend preprocesses the text, transforms it using the saved TF-IDF vectorizer, and sends it through the saved Logistic Regression model.

The response contains:

- Predicted sentiment
- Probability for negative
- Probability for neutral
- Probability for positive

---

## Running the Frontend

From the project root:

```powershell
py app.py
```

You should see:

```text
Model and TF-IDF vectorizer loaded successfully.
Open http://127.0.0.1:5000
```

Open the following in your browser:

```text
http://127.0.0.1:5000
```

### Important

Do **not** run:

```text
frontend/index.html
```

directly with VS Code Live Server if you want the complete application.

Opening the HTML directly bypasses Flask, so the `/api/predict` backend is not available.

The correct method is:

```text
py app.py
       ↓
http://127.0.0.1:5000
```

---

## Example

Input:

```text
I absolutely love this product!
```

The application sends the text through:

```text
Input
 ↓
NLTK preprocessing
 ↓
TF-IDF
 ↓
Logistic Regression
 ↓
Prediction
```

Possible result:

```text
Predicted Sentiment: positive
```

The dashboard also displays the model's probability estimate for each class.

---

## Model Limitation

The frontend displays the output of the existing classical ML model. Therefore, an incorrect prediction is not necessarily a frontend/API error.

For example:

```text
I didn't like that movie.
```

may occasionally be classified as neutral.

The current model has approximately:

```text
Accuracy: 65.71%
Macro F1: 0.66
```

The model uses TF-IDF + Logistic Regression, so it learns statistical word patterns rather than full contextual language understanding.

The current preprocessing also removes punctuation, which means contractions can become:

```text
didn't → didnt
can't   → cant
wasn't  → wasnt
```

Improving negation and contraction handling would require changing the NLP pipeline and retraining the model.

---

## Why the Frontend Uses the Saved Model

The frontend is connected to:

```text
models/sentiment_model.pkl
models/tfidf_vectorizer.pkl
```

This means the browser application uses the same trained model and vectorizer produced by the project's training pipeline.

There is no separate frontend-only model.

---

## Requirements

The frontend requires Flask in addition to the NLP/ML libraries:

```text
pandas
scikit-learn
nltk
joblib
matplotlib
seaborn
flask
```

Install them with:

```powershell
py -m pip install pandas scikit-learn nltk joblib matplotlib seaborn flask
```

If NLTK resources are not installed:

```powershell
py -c "import nltk; nltk.download('punkt'); nltk.download('punkt_tab'); nltk.download('stopwords'); nltk.download('wordnet'); nltk.download('averaged_perceptron_tagger'); nltk.download('averaged_perceptron_tagger_eng')"
```

---

## Result

The frontend provides a complete demonstration layer for the NLP project:

```text
Machine Learning Model
        +
Flask Backend
        +
HTML/CSS/JavaScript
        =
Interactive Sentiment Analysis Application
```

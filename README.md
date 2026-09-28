# Intelligent Multi-Class Natural Language Text Sentiment Classifier

An end-to-end Natural Language Processing (NLP) and Machine Learning project that classifies English text into three sentiment categories:

- **Negative**
- **Neutral**
- **Positive**

The project covers the complete workflow from raw text preprocessing and feature extraction to model training, evaluation, prediction, and an interactive Flask web application.

---

## Project Overview

This project was developed as an NLP / AI-ML internship project and demonstrates how unstructured natural-language text can be converted into numerical features and classified using a supervised machine-learning model.

### Pipeline

```text
Raw Text
   ↓
Text Preprocessing
   ↓
Tokenization
   ↓
Stopword Removal
   ↓
POS Tagging
   ↓
WordNet POS Mapping
   ↓
Lemmatization
   ↓
TF-IDF Vectorization
   ↓
Logistic Regression
   ↓
Sentiment Prediction
   ↓
Evaluation
   ↓
Flask Web Application
```

---

## Features

- Multi-class sentiment classification
- Three sentiment classes: negative, neutral, positive
- NLTK-based text preprocessing
- Lowercasing
- Punctuation removal
- Tokenization
- Stopword removal
- Part-of-Speech (POS) tagging
- WordNet POS mapping
- POS-aware lemmatization
- TF-IDF feature extraction
- Logistic Regression classifier
- Train/test split with stratification
- Accuracy, precision, recall and F1-score evaluation
- Confusion matrix
- Prediction probabilities
- Saved trained model and TF-IDF vectorizer using Joblib
- Interactive Flask web application
- Responsive HTML/CSS/JavaScript frontend
- API endpoint for predictions

---

## Dataset

The project uses a multi-class sentiment dataset containing **31,232 text records**.

### Classes

| Class | Meaning |
|---|---|
| Negative | Text expressing an unfavorable or negative sentiment |
| Neutral | Text with neutral, factual, or less clearly emotional sentiment |
| Positive | Text expressing a favorable or positive sentiment |

### Dataset columns

- `id` — record identifier
- `text` — original input text
- `label` — original numeric class
- `sentiment` — sentiment class

### Label mapping

```text
0 → negative
1 → neutral
2 → positive
```

The dataset is stored locally at:

```text
data/dataset.csv
```

---

## Text Preprocessing

The preprocessing pipeline is implemented in:

```text
src/preprocessing.py
```

### 1. Lowercasing

```python
text = text.lower()
```

Example:

```text
"I LOVE THIS!"
        ↓
"i love this!"
```

### 2. Punctuation Removal

```python
text = text.translate(
    str.maketrans("", "", string.punctuation)
)
```

Example:

```text
"Excellent!!!"
      ↓
"Excellent"
```

### 3. Tokenization

```python
tokens = word_tokenize(text)
```

Example:

```text
"I love this movie."
        ↓
["I", "love", "this", "movie", "."]
```

### 4. Stopword Removal

```python
stop_words = set(stopwords.words("english"))

filtered_tokens = []

for token in tokens:
    if token not in stop_words:
        filtered_tokens.append(token)
```

### 5. POS Tagging

```python
pos_tags = pos_tag(filtered_tokens)
```

POS tagging identifies grammatical roles such as nouns, verbs, adjectives, and adverbs.

### 6. WordNet POS Mapping

NLTK POS tags are converted into WordNet-compatible tags:

```text
J → adjective
V → verb
N → noun
R → adverb
```

### 7. Lemmatization

```python
lemmatized_token = lemmatizer.lemmatize(
    token,
    get_wordnet_pos(tag)
)
```

Example:

```text
"loved" → "love"
"running" → "run"
"cars" → "car"
```

Example complete transformation:

```text
"I absolutely LOVED this movie!!!"
                ↓
"absolutely love movie"
```

---

## TF-IDF Feature Extraction

TF-IDF (Term Frequency-Inverse Document Frequency) converts cleaned text into numerical features that can be used by the machine-learning model.

```python
vectorizer = TfidfVectorizer()

X_train = vectorizer.fit_transform(X_train_text)
X_test = vectorizer.transform(X_test_text)
```

### Important

The vectorizer is fitted **only on the training data**:

```python
fit_transform(X_train_text)
```

The test data uses:

```python
transform(X_test_text)
```

This prevents test-data leakage.

### Project dimensions

```text
Training samples: 24,985
Testing samples: 6,247

TF-IDF training shape: (24985, 25182)
TF-IDF testing shape:  (6247, 25182)
```

---

## Model

The classifier is:

**Logistic Regression**

```python
from sklearn.linear_model import LogisticRegression

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)
```

Logistic Regression is a common and effective baseline for text classification, especially when combined with TF-IDF features.

---

## Evaluation

The trained model was evaluated on **6,247 unseen test samples**.

### Results

| Metric | Result |
|---|---:|
| Accuracy | **65.71%** |
| Macro F1-score | **0.66** |
| Weighted F1-score | **0.66** |

### Per-class performance

| Class | Precision | Recall | F1-score |
|---|---:|---:|---:|
| Negative | 0.66 | 0.60 | 0.63 |
| Neutral | 0.59 | 0.65 | 0.62 |
| Positive | 0.74 | 0.72 | 0.73 |

### Confusion Matrix

```text
                 Predicted
              Neg   Neu   Pos

Actual Neg   1088   600   133
Actual Neu    421  1514   395
Actual Pos    128   465  1503
```

The diagonal values represent correct predictions.

---

## Prediction

The project includes an interactive prediction script:

```text
src/predict.py
```

Example:

```text
Enter a sentence: I absolutely love this product!

Predicted Sentiment: positive

Prediction Probabilities:
negative: 3.17%
neutral: 2.23%
positive: 94.60%
```

### Important distinction

A prediction probability such as:

```text
positive: 94.60%
```

is the model's probability estimate for **that individual input**.

It is **not** the same thing as the model's test accuracy.

The overall test accuracy of this project is approximately:

```text
65.71%
```

---

## Known Model Limitations

This project is a classical TF-IDF + Logistic Regression baseline. It does not understand language in the same way as a modern large language model or transformer-based NLP system.

For example, a sentence such as:

```text
"I didn't like that movie."
```

can sometimes be misclassified.

This can happen because:

- TF-IDF represents word statistics rather than full semantic meaning.
- Logistic Regression learns statistical relationships from the training data.
- Negation and context can be difficult for a bag-of-words-style representation.
- The model has approximately 65.71% test accuracy, so incorrect predictions are expected.
- The current preprocessing removes punctuation, so contractions such as `didn't` become `didnt`.

These limitations are useful observations rather than frontend errors. Improving them would require changes such as better contraction/negation handling, n-grams, hyperparameter tuning, a larger or improved dataset, or a more advanced NLP model.

---

## Saved Model Files

The trained artifacts are stored in:

```text
models/
├── sentiment_model.pkl
├── tfidf_vectorizer.pkl
└── test_data.pkl
```

They can be loaded using:

```python
import joblib

model = joblib.load("models/sentiment_model.pkl")
vectorizer = joblib.load("models/tfidf_vectorizer.pkl")
```

---

## Web Application

The project also contains an interactive web application built with:

- Flask
- HTML
- CSS
- JavaScript
- Existing trained Logistic Regression model
- Existing TF-IDF vectorizer

### Web application workflow

```text
User enters text
      ↓
JavaScript sends request
      ↓
Flask API receives text
      ↓
NLTK preprocessing
      ↓
Saved TF-IDF vectorizer
      ↓
Saved Logistic Regression model
      ↓
Prediction + probabilities
      ↓
JSON response
      ↓
Frontend displays result
```

### Frontend features

- Text input area
- Character counter
- Positive / neutral / negative example buttons
- Analyze Sentiment button
- Predicted sentiment display
- Probability bars for all three classes
- Model information
- NLP pipeline visualization
- Project performance metrics
- Responsive dashboard layout

The frontend documentation is available in:

```text
README_FRONTEND_SECTION.md
```

---

## Flask API

The application exposes prediction functionality through a Flask endpoint.

The frontend sends the user's text to the backend, which returns the predicted sentiment and class probabilities.

The application can be started with:

```powershell
py app.py
```

Then open:

```text
http://127.0.0.1:5000
```

Do **not** open the HTML file directly with Live Server for the full application.

The HTML frontend depends on the Flask backend and its `/api/predict` endpoint.

---

## Project Structure

```text
Intelligent-Multi-Class-Natural-Language-Text-Sentiment-Classifier/
│
├── data/
│   └── dataset.csv
│
├── src/
│   ├── preprocessing.py
│   ├── train.py
│   ├── evaluate.py
│   └── predict.py
│
├── models/
│   ├── sentiment_model.pkl
│   ├── tfidf_vectorizer.pkl
│   └── test_data.pkl
│
├── results/
│   ├── classification_report.txt
│   ├── confusion_matrix.png
│   └── metrics.txt
│
├── reports/
│   └── Task_2_Report.pdf
│
├── frontend/
│   ├── index.html
│   ├── script.js
│   └── style.css
│
├── app.py
├── requirements.txt
├── requirements_frontend.txt
├── README.md
├── README_FRONTEND_SECTION.md
└── LICENSE
```

---

## Installation

Install the required Python packages:

```powershell
py -m pip install pandas scikit-learn nltk joblib matplotlib seaborn flask
```

Download the required NLTK resources:

```powershell
py -c "import nltk; nltk.download('punkt'); nltk.download('punkt_tab'); nltk.download('stopwords'); nltk.download('wordnet'); nltk.download('averaged_perceptron_tagger'); nltk.download('averaged_perceptron_tagger_eng')"
```

---

## Running the Project

### Run the trained model prediction script

From the project root:

```powershell
py src\predict.py
```

### Run evaluation

```powershell
py src\evaluate.py
```

### Run the web application

```powershell
py app.py
```

Open:

```text
http://127.0.0.1:5000
```

---

## Technologies Used

- Python
- NLTK
- Pandas
- NumPy
- Scikit-learn
- Joblib
- Matplotlib
- Seaborn
- Flask
- HTML
- CSS
- JavaScript

---

## Learning Outcomes

This project demonstrates practical understanding of:

- Natural Language Processing
- Text preprocessing
- Tokenization
- Stopword removal
- POS tagging
- Lemmatization
- TF-IDF
- Supervised machine learning
- Logistic Regression
- Multiclass classification
- Model evaluation
- F1-score
- Confusion matrices
- Model persistence
- REST-style API communication
- Flask web integration
- Frontend/backend integration

---

## Project Status

**Completed**

The core NLP classifier, evaluation pipeline, saved model artifacts, and interactive Flask frontend are implemented and working.

The current model is intentionally a classical ML baseline. Future improvements could focus on handling negation and contractions, tuning TF-IDF/model parameters, experimenting with n-grams, or using transformer-based NLP models.

---

## License

This project is licensed under the MIT License. See `LICENSE` for details.

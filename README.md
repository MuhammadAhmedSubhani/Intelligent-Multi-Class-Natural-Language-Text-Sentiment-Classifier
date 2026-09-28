# Intelligent Multi-Class Natural Language Text Sentiment Classifier

## Project Overview

This project is a multi-class Natural Language Processing (NLP) sentiment classification system developed as part of an AI/ML internship task.

The system takes an input sentence and classifies it into one of three sentiment categories:

* Negative
* Neutral
* Positive

The project uses Natural Language Processing techniques for text preprocessing, TF-IDF for converting text into numerical features, and Logistic Regression for sentiment classification.

---

## Project Objectives

The main objectives of this project are to:

* Preprocess natural language text.
* Remove unnecessary words and punctuation.
* Apply POS-aware lemmatization.
* Convert text into numerical representations using TF-IDF.
* Train a multi-class sentiment classification model.
* Evaluate the model using Accuracy and F1-score.
* Visualize classification performance using a confusion matrix.
* Predict the sentiment of new user-provided sentences.

---

## Dataset

The project uses the **Multiclass Sentiment Analysis Dataset**.

The dataset contains:

* **31,232 text samples**
* **4 columns**
* Three sentiment classes:

  * Negative
  * Neutral
  * Positive

The dataset is stored locally in:

```text
data/dataset.csv
```

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
├── requirements.txt
└── README.md
```

---

## Text Preprocessing

The text preprocessing pipeline is implemented in `src/preprocessing.py`.

The following techniques are applied:

1. Lowercasing
2. Punctuation removal
3. Tokenization
4. Stopword removal
5. Part-of-Speech (POS) tagging
6. WordNet POS mapping
7. POS-aware lemmatization

For example:

```text
"I absolutely LOVED this movie!!!"
```

is transformed into:

```text
"absolutely love movie"
```

---

## Feature Extraction

After preprocessing, the text is converted into numerical features using **TF-IDF (Term Frequency-Inverse Document Frequency)**.

The vectorizer is fitted only on the training data to avoid test-data leakage.

The resulting feature dimensions are:

```text
Training data: (24985, 25182)
Testing data:  (6247, 25182)
```

---

## Model

The classification model used is **Logistic Regression** from Scikit-Learn.

The model is configured with:

```python
LogisticRegression(max_iter=1000)
```

The model is trained using the TF-IDF features and the corresponding sentiment labels.

---

## Train/Test Split

The dataset is divided into:

* **80% training data:** 24,985 samples
* **20% testing data:** 6,247 samples

A fixed `random_state=42` is used to make the split reproducible.

Stratification is also used to preserve the distribution of sentiment classes.

---

## Model Evaluation

The trained model was evaluated using:

* Accuracy
* Precision
* Recall
* F1-score
* Confusion Matrix

### Overall Results

| Metric            | Result |
| ----------------- | -----: |
| Accuracy          | 65.71% |
| Macro F1-score    |   0.66 |
| Weighted F1-score |   0.66 |

### Class-wise F1-score

| Sentiment | F1-score |
| --------- | -------: |
| Negative  |     0.63 |
| Neutral   |     0.62 |
| Positive  |     0.73 |

---

## Confusion Matrix

The confusion matrix is stored at:

```text
results/confusion_matrix.png
```

The matrix is:

```text
[[1088, 600, 133],
 [421, 1514, 395],
 [128, 465, 1503]]
```

Rows represent the actual sentiment and columns represent the predicted sentiment.

---

## Prediction

The `src/predict.py` script allows the user to enter a new sentence and receive:

* Predicted sentiment
* Probability for each sentiment class

Example:

```text
Enter a sentence: I absolutely love this product!

Predicted Sentiment: positive

Prediction Probabilities:
negative: 3.17%
neutral: 2.23%
positive: 94.60%
```

---

## How to Run

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Train the model

From the `src` directory:

```bash
python train.py
```

This creates the trained model, TF-IDF vectorizer, and test data inside the `models` directory.

### 3. Evaluate the model

```bash
python evaluate.py
```

This generates the classification report, metrics file, and confusion matrix.

### 4. Make predictions

```bash
python predict.py
```

Enter a sentence when prompted.

---

## Technologies Used

* Python
* NLTK
* Pandas
* Scikit-Learn
* Joblib
* Matplotlib
* Seaborn
* TF-IDF
* Logistic Regression

---

## Project Outcome

The project successfully implements an end-to-end multi-class sentiment classification pipeline, starting from raw text preprocessing and feature extraction and continuing through model training, evaluation, visualization, and prediction on new text.

The trained model achieved **65.71% accuracy** and a **0.66 macro F1-score** on the held-out test set.

---

## Internship Task

This project was developed as part of an **AI/ML internship task focused on Natural Language Processing and multi-class sentiment classification**.

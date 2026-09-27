# Multi-Class Natural Language Text Sentiment Classifier

## Project objective

This project classifies short text messages into one of three sentiment
categories:

- `negative`
- `neutral`
- `positive`

The pipeline cleans the text, converts it into TF-IDF features, trains a
Logistic Regression classifier, evaluates the classifier, and supports
predictions for new user-provided sentences.

## Project structure

```text
NLP-Sentiment-Classifier/
├── data/
│   └── dataset.csv
├── models/
│   ├── sentiment_model.pkl
│   ├── tfidf_vectorizer.pkl
│   └── test_data.pkl
├── results/
│   ├── classification_report.txt
│   ├── confusion_matrix.png
│   └── metrics.txt
├── src/
│   ├── preprocessing.py
│   ├── train.py
│   ├── evaluate.py
│   └── predict.py
├── requirements.txt
├── LICENSE
└── README.md
```

## Dataset

The project uses the local [`data/dataset.csv`](data/dataset.csv) file. It
contains 31,232 labelled text records with these columns:

| Column | Description |
| --- | --- |
| `id` | Record identifier |
| `text` | Original text to classify |
| `label` | Numeric label supplied with the dataset |
| `sentiment` | Target class used for training |

The sentiment distribution is:

| Sentiment | Records |
| --- | ---: |
| Negative | 9,105 |
| Neutral | 11,649 |
| Positive | 10,478 |

## Text preprocessing

[`src/preprocessing.py`](src/preprocessing.py) applies the following steps:

1. Convert text to lowercase.
2. Remove punctuation.
3. Tokenize the text with NLTK.
4. Remove English stopwords.
5. Apply part-of-speech tagging.
6. Lemmatize tokens using the POS-aware WordNet lemmatizer.

The required NLTK resources are downloaded automatically when the preprocessing
module is imported.

## TF-IDF feature extraction

[`src/train.py`](src/train.py) uses scikit-learn's
`TfidfVectorizer` to represent the cleaned text numerically. The vectorizer is
fit only on the training text, and the test text is transformed using that
same learned vocabulary. The fitted vectorizer is saved as
`models/tfidf_vectorizer.pkl`.

## Logistic Regression classifier

The classifier is scikit-learn's `LogisticRegression` with `max_iter=1000`.
It learns the relationship between the TF-IDF features and the three sentiment
classes. The trained model is saved as `models/sentiment_model.pkl`.

## Train/test split

The dataset is split into 80% training data and 20% test data using
`train_test_split` with:

- `random_state=42` for reproducibility
- `stratify=y` to preserve the sentiment distribution

This produces 24,985 training records and 6,247 test records. The test
features and labels are saved to `models/test_data.pkl` for evaluation.

## Evaluation metrics

Run [`src/evaluate.py`](src/evaluate.py) after training to calculate accuracy,
the per-class precision/recall/F1 report, macro F1-score, weighted F1-score,
and a confusion matrix.

The currently saved results are:

| Metric | Score |
| --- | ---: |
| Accuracy | 0.6571 |
| Macro F1-score | 0.6582 |
| Weighted F1-score | 0.6579 |

Per-class F1-scores from `results/classification_report.txt`:

| Class | Precision | Recall | F1-score | Support |
| --- | ---: | ---: | ---: | ---: |
| Negative | 0.66 | 0.60 | 0.63 | 1,821 |
| Neutral | 0.59 | 0.65 | 0.62 | 2,330 |
| Positive | 0.74 | 0.72 | 0.73 | 2,096 |

The full report is in
[`results/classification_report.txt`](results/classification_report.txt),
the aggregate metrics are in
[`results/metrics.txt`](results/metrics.txt), and the visual confusion matrix
is in [`results/confusion_matrix.png`](results/confusion_matrix.png).

## How to run the project

Run these commands from the project root.

### 1. Install dependencies

```bash
python -m pip install -r requirements.txt
```

### 2. Train the model

```bash
python src/train.py
```

This preprocesses the dataset, trains the classifier, and creates the files in
the `models/` directory.

### 3. Evaluate the model

```bash
python src/evaluate.py
```

This updates the files in the `results/` directory.

### 4. Predict a new sentence

```bash
python src/predict.py
```

Enter a sentence when prompted. The script prints the predicted sentiment and
the probability assigned to each class.

## Example prediction

Example input:

```text
I absolutely loved this product!
```

Example output from the saved model:

```text
Predicted Sentiment: positive

Prediction Probabilities:
negative: 3.17%
neutral: 2.23%
positive: 94.60%
```

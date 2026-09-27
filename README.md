NLP-Sentiment-Classifier/
│
├── data/
│   └── dataset.csv
│
├── src/
│   ├── preprocessing.py   ← Text cleaning
│   ├── train.py           ← TF-IDF + model training
│   ├── evaluate.py        ← Metrics + confusion matrix
│   └── predict.py         ← New-text predictions
│
├── models/
│   └── sentiment_model.pkl
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


# Table with file and what it is suppose to do


| What we're doing                   | File                   |
| ---------------------------------- | ---------------------- |
| Text cleaning functions            | `src/preprocessing.py` |
| Loading dataset for training       | `src/train.py`         |
| TF-IDF                             | `src/train.py`         |
| Train Logistic Regression          | `src/train.py`         |
| Save trained model                 | `src/train.py`         |
| Accuracy / Precision / Recall / F1 | `src/evaluate.py`      |
| Confusion matrix                   | `src/evaluate.py`      |
| Predict new sentences              | `src/predict.py`       |
| Final metrics                      | `results/`             |
| Final internship report            | `reports/`             |

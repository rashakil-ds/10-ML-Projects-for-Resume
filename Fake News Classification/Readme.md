
# Fake vs Real News Classification (NLP + TF-IDF + Logistic Regression)

This project implements an end-to-end **text classification pipeline** to classify text as **Fake (0)** or **Real (1)** using traditional Machine Learning techniques and clean NLP preprocessing. It includes dataset exploration, preprocessing, TF-IDF vectorization, multiple model training, evaluation, and a production-ready prediction system with model + vectorizer saving.

---

## Project Overview
### Steps
- Exploratory Data Analysis (EDA)
- Text cleaning and preprocessing
- Lemmatization using WordNetLemmatizer
- Train–test split 70% and 30%
- TF-IDF vectorization on training data only
- Model training:
  - Multinomial Naive Bayes
  - Bernoulli Naive Bayes
  - Logistic Regression
- Evaluation: accuracy, confusion matrix, classification report, ROC curves
- Model comparison with bar charts
- Saving and loading:
  - Trained model
  - TF-IDF vectorizer
- Production-ready inference function
---
## Dataset

[The dataset](https://www.kaggle.com/datasets/studymart/welfake-dataset-for-fake-news) contains three columns:

| Column | Description |
|--------|-------------|
| `title` | Small title |
| `text` | Raw text content |
| `label` | 0 = Fake, 1 = Real |

The dataset was cleaned, null values replaced, and text normalized.

---
## Text Preprocessing Steps
1. Remove punctuation  
2. Remove stopwords  
3. Convert text to lowercase  
4. Tokenize  
5. Lemmatize each token  
6. Reconstruct cleaned text
This ensures the model trains on high-quality and normalized input.
---
## Model Training
Three models were trained and evaluated:
- **Multinomial Naive Bayes (MNB)**
- **Bernoulli Naive Bayes (BNB)**
- **Logistic Regression (LR)** ← Best performing

TF-IDF vectorization was applied **only on training data** to avoid data leakage.
Performance was compared using:
- Accuracy
- Confusion Matrix
- Classification Report
- ROC-AUC curves

---

## Best Model

**Logistic Regression** achieved the highest performance and generalization capability.  
This model was selected for production use.

---

## Model & Vectorizer Saving

Both the trained Logistic Regression model and TF-IDF vectorizer are saved together:

```python
with open("logistic_regression_tfidf.pkl", "wb") as f:
    pickle.dump({
        "model": lr_model,
        "vectorizer": vectorizer
    }, f)

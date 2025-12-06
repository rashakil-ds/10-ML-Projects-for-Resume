import streamlit as st
import pickle
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

nltk.download("stopwords", quiet=True)
nltk.download("wordnet", quiet=True)

#loading trained model
@st.cache_resource
def load_model_and_vectorizer():
    with open("logistic_regression_tfidf.pkl", "rb") as f:
        saved_objects = pickle.load(f)

    loaded_LR = saved_objects["model"]
    loaded_vectorizer = saved_objects["vectorizer"]
    return loaded_LR, loaded_vectorizer


loaded_LR, loaded_vectorizer = load_model_and_vectorizer()
english_punctuation = string.punctuation
lemmatizer = WordNetLemmatizer()

#Try to load NLTK stopwords; if something is wrong, fall back to a manual list
try:
    english_stopwords = set(stopwords.words("english"))
except Exception:
    english_stopwords = {
        "the", "a", "an", "is", "am", "are", "and", "or", "if", "to", "of", "in",
        "on", "for", "with", "this", "that", "it", "at", "as", "be", "by", "from",
        "about", "was", "were", "will", "would", "can", "could", "should", "have",
        "has", "had", "do", "does", "did", "not", "no", "so", "but", "than", "then"
    }

def preprocess_text(text: str) -> str:

    #remove punctuation
    remove_punc = [char for char in text if char not in english_punctuation]
    clean_text = "".join(remove_punc)

    #remove stopwords
    words = clean_text.split()
    filtered_words = [word for word in words if word.lower() not in english_stopwords]
    return " ".join(filtered_words)


def lemmatize_text(text: str) -> str:
    return " ".join([lemmatizer.lemmatize(word) for word in text.split()])


def predict_for_text(random_text: str):

    # same pipeline as in notebook
    preprocessed_text = preprocess_text(random_text)
    lemmatized_text = lemmatize_text(preprocessed_text)
    text_vector = loaded_vectorizer.transform([lemmatized_text])

    prediction = loaded_LR.predict(text_vector)[0]

    prediction_proba = None
    if hasattr(loaded_LR, "predict_proba"):
        #probability of class 1 (Real)
        prediction_proba = loaded_LR.predict_proba(text_vector)[0, 1]

    return prediction, prediction_proba


#Streamlit UI
st.title("Fake vs. Real News Classifier")
st.write(
    "Paste any news text or message below. The model will predict whether it is "
    "**Fake News (0)** or **Real News (1)** using a Logistic Regression classifier trained on TF-IDF features."
)

user_input = st.text_area("Enter text to classify:", height=300)

if st.button("Make Prediction"):

    if not user_input.strip():
        st.warning("Please enter some text first.")
    else:
        label, proba = predict_for_text(user_input)

        if label == 1:
            pred_label = "Real News (Class 1)"
            icon = "✅"
        else:
            pred_label = "Fake News (Class 0)"
            icon = "⚠️"

        st.markdown(f"### {icon} Prediction: **{pred_label}**")

        if proba is not None:
            st.write(
                f"Model confidence (probability of *Real* is:): **{proba:.4f}**"
            )

st.caption("• Built by @Rashedul Alam")

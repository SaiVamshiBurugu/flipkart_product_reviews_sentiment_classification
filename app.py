import streamlit as st
import joblib
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Set page config
st.set_page_config(page_title="Badminton Review Sentiment Classifier", page_icon="🏸")

# Download NLTK data
@st.cache_resource
def download_nltk_data():
    nltk.download('stopwords')
    nltk.download('wordnet')
    nltk.download('omw-1.4')

download_nltk_data()

# Load the model
@st.cache_resource
def load_model():
    return joblib.load('sentiment_pipeline.joblib')

pipeline = load_model()

# Preprocessing function (must match training)
stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"#", "", text)
    text = re.sub(r"[^a-z\s]", "", text)
    words = text.split()
    words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]
    return " ".join(words)

# UI
st.title("🏸 Badminton Review Sentiment Classifier")
st.write("Enter a review for a badminton product to predict its sentiment.")

review_text = st.text_area("Review Text", height=150)

if st.button("Predict Sentiment"):
    if review_text:
        cleaned_review = clean_text(review_text)
        prediction = pipeline.predict([cleaned_review])[0]
        
        # Map prediction to label
        # 0: Negative, 1: Neutral, 2: Positive
        sentiment_map = {0: "Negative", 1: "Neutral", 2: "Positive"}
        result = sentiment_map.get(prediction, "Unknown")
        
        st.subheader("Prediction:")
        if result == "Positive":
            st.success(f"The sentiment is **{result}** 😊")
        elif result == "Negative":
            st.error(f"The sentiment is **{result}** 😞")
        else:
            st.warning(f"The sentiment is **{result}** 😐")

    else:
        st.warning("Please enter some text.")

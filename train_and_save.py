import pandas as pd
import numpy as np
import re
import joblib
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split

# Ensure NLTK data is downloaded
nltk.download('stopwords')
nltk.download('wordnet')

# Load data
df = pd.read_csv("batminton_data.csv")

# Preprocessing
stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

def clean_text(text):
    if pd.isna(text):
        return ""
    text = text.lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"#", "", text)
    text = re.sub(r"[^a-z\s]", "", text)
    words = text.split()
    words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]
    return " ".join(words)

# Clean data
df = df.dropna(subset=['Review text', 'Ratings'])
df = df.drop_duplicates(subset=['Review text'])
df["cleaned_review"] = df["Review text"].apply(clean_text)

# Map ratings to sentiment
# Note: Notebook mapped {1:"Negative", 2:"Negative", 3:"Neutral", 4:"Positive", 5:"Positive"}
# Then mapped {"Negative":"0", "Neutral":"1", "Positive":"2"} and cast to int.
# So: 1,2 -> 0; 3 -> 1; 4,5 -> 2
def map_sentiment(rating):
    if rating in [1, 2]:
        return 0
    elif rating == 3:
        return 1
    elif rating in [4, 5]:
        return 2
    return -1

df["Ratings"] = df["Ratings"].apply(map_sentiment)
df = df[df["Ratings"] != -1]

X = df["cleaned_review"]
y = df["Ratings"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

# Create and train pipeline
pipe = Pipeline([
    ('count_vectorizer', CountVectorizer()),
    ('rf_classifier', RandomForestClassifier(n_jobs=-1))
])

print("Training pipeline...")
pipe.fit(X_train, y_train)

# Save pipeline
joblib.dump(pipe, 'sentiment_pipeline.joblib')
print("Pipeline saved to sentiment_pipeline.joblib")

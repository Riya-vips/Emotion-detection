import re
import string
import pandas as pd
import neattext.functions as nfx
import nltk

from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder

nltk.download('wordnet')
nltk.download('omw-1.4')


class DataTransformation:
    def __init__(self):
        self.lemmatizer = WordNetLemmatizer()

    def clean_text(self, text):
        text = text.lower()
        text = re.sub(r'https?://\S+|www\.\S+', '', text)
        text = re.sub(r'<.*?>+', '', text)
        text = re.sub(r'[%s]' % re.escape(string.punctuation), '', text)
        text = re.sub(r'\n', ' ', text)
        text = re.sub(r'\w*\d\w*', '', text)
        text = re.sub(r'[^\x00-\x7f]', '', text)
        return " ".join([self.lemmatizer.lemmatize(word) for word in text.split()])

    def transform(self, df):
        df.columns = df.columns.str.lower().str.strip()

        if 'text' not in df.columns or 'emotion' not in df.columns:
            raise ValueError("DataFrame must contain 'text' and 'emotion' columns.")

        df['clean_text'] = df['text'].astype(str).apply(nfx.remove_userhandles)
        df['clean_text'] = df['clean_text'].apply(self.clean_text)
        df['clean_text'] = df['clean_text'].apply(nfx.remove_stopwords)

        X = df['clean_text']
        y = df['emotion']

        label_encoder = LabelEncoder()
        y_encoded = label_encoder.fit_transform(y)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
        )

        vectorizer = TfidfVectorizer(
            max_features=6000,
            ngram_range=(1, 2),
            sublinear_tf=True,
            lowercase=True,
            stop_words='english'
        )

        X_train_vec = vectorizer.fit_transform(X_train)
        X_test_vec = vectorizer.transform(X_test)

        return X_train_vec, X_test_vec, y_train, y_test, vectorizer, label_encoder

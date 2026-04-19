import pandas as pd
import re
import string

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Download required NLTK data (run once)
nltk.download("stopwords")
nltk.download("wordnet")
nltk.download("omw-1.4")


class TextPreprocessor:
    def __init__(self):
        self.stop_words = set(stopwords.words("english"))
        self.lemmatizer = WordNetLemmatizer()

    def clean_text(self, text: str) -> str:
        """
        Basic text cleaning:
        - lowercase
        - remove punctuation
        - remove numbers
        - remove extra spaces
        """
        if pd.isna(text):
            return ""

        text = text.lower()
        text = re.sub(r"\d+", "", text)  # remove numbers
        text = text.translate(str.maketrans("", "", string.punctuation))
        text = re.sub(r"\s+", " ", text).strip()

        return text

    def tokenize_and_lemmatize(self, text: str) -> str:
        """
        Remove stopwords + lemmatize
        """
        tokens = text.split()

        tokens = [
            self.lemmatizer.lemmatize(word)
            for word in tokens
            if word not in self.stop_words
        ]

        return " ".join(tokens)

    def preprocess_text(self, text: str) -> str:
        """
        Full NLP preprocessing pipeline
        """
        text = self.clean_text(text)
        text = self.tokenize_and_lemmatize(text)
        return text

    def process_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply preprocessing to Likes and Dislikes
        """

        # Clean individual columns
        df["likes_clean"] = df["Likes"].apply(self.preprocess_text)
        df["dislikes_clean"] = df["Dislikes"].apply(self.preprocess_text)

        # Combined text (VERY important for chatbot + topics)
        df["combined_text"] = (
            df["likes_clean"].fillna("") + " " + df["dislikes_clean"].fillna("")
        )

        return df
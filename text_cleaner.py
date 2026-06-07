import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

from sklearn.base import BaseEstimator, TransformerMixin

import nltk

def download_nltk_resources():
    resources = [
        ('tokenizers/punkt', 'punkt'),
        ('tokenizers/punkt_tab', 'punkt_tab'),
        ('corpora/stopwords', 'stopwords'),
        ('corpora/wordnet', 'wordnet')
    ]

    for path, name in resources:
        try:
            nltk.data.find(path)
        except LookupError:
            nltk.download(name)

download_nltk_resources()

stop_words = set(stopwords.words("english"))
lemma = WordNetLemmatizer()


def clean_text(text):

    text = re.sub("[^a-zA-Z]", " ", text)
    text = text.lower()

    words = word_tokenize(text)

    words = [lemma.lemmatize(word) for word in words if word not in stop_words]

    return " ".join(words)


class TextCleaner(BaseEstimator, TransformerMixin):

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return [clean_text(text) for text in X]
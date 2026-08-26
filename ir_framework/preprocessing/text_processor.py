import re
from typing import List
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.corpus import stopwords


class TextProcessor:
    """
    Handles text processing operations including tokenization, stop word removal, stemming, and lemmatization.
    Maintains exact parity with original ir_lib functions.
    """

    def __init__(self, mode: str = "none", language: str = "english"):
        self.mode = mode
        self.language = language
        self.ps = PorterStemmer()
        self._lemmatizer = None
        self._stopwords = None

    @property
    def lemmatizer(self) -> WordNetLemmatizer:
        if self._lemmatizer is None:
            self._lemmatizer = WordNetLemmatizer()
        return self._lemmatizer

    @property
    def stop_words(self) -> set:
        if self._stopwords is None:
            self._stopwords = set(stopwords.words(self.language))
        return self._stopwords

    def tokenize(self, text: str) -> List[str]:
        """Extracts alphabetic tokens in lowercase and sorts them."""
        tokens = re.findall(r"[a-zA-Z]+", text.lower())
        tokens.sort()
        return tokens

    def remove_stopwords(self, vocabulary: List[str]) -> List[str]:
        """Removes stop words from vocabulary list."""
        stop_words_set = self.stop_words
        return [word for word in vocabulary if word not in stop_words_set]

    def stem(self, vocabulary: List[str]) -> List[str]:
        """Stems words using PorterStemmer and returns unique sorted stems."""
        stemmed = [self.ps.stem(word) for word in vocabulary]
        return sorted(set(stemmed))

    def lemmatize(self, vocabulary: List[str]) -> List[str]:
        """Lemmatizes words using WordNetLemmatizer and returns unique sorted lemmas."""
        lemmatized = [self.lemmatizer.lemmatize(word) for word in vocabulary]
        return sorted(set(lemmatized))

    def process_tokens(self, tokens: List[str], mode: str = None) -> List[str]:
        """Applies preprocessing mode (stemming, lemmatization, no_stop_words, both, or none) to tokens."""
        effective_mode = mode if mode is not None else self.mode

        if effective_mode == "stemming":
            return self.stem(tokens)
        elif effective_mode == "lemmatization":
            return [self.lemmatizer.lemmatize(word) for word in tokens]
        elif effective_mode == "no_stop_words":
            return self.remove_stopwords(tokens)
        elif effective_mode == "both":
            return self.stem(self.remove_stopwords(tokens))
        return tokens

    def process_text(self, text: str, mode: str = None) -> List[str]:
        """Tokenizes text and applies preprocessing mode."""
        tokens = self.tokenize(text)
        return self.process_tokens(tokens, mode=mode)


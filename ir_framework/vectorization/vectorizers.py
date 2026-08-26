import re
import numpy as np
from abc import ABC, abstractmethod
from typing import List, Dict, Tuple, Union, Optional
from collections import Counter

from ir_framework.data.dataset import IRDataset, QueryDataset, Query
from ir_framework.indexing.vocabulary import Vocabulary
from ir_framework.preprocessing.text_processor import TextProcessor


class BaseVectorizer(ABC):
    """Abstract Base Class for Document and Query Vectorizers."""

    @abstractmethod
    def transform_documents(
        self,
        dataset: IRDataset,
        vocabulary: Vocabulary,
        text_column: Optional[str] = None
    ) -> List[np.ndarray]:
        pass

    @abstractmethod
    def transform_queries(
        self,
        queries: Union[QueryDataset, List[Tuple[str, str]], List[str]],
        vocabulary: Vocabulary,
        pre_processing_mode: str = "none"
    ) -> List[np.ndarray]:
        pass


class CountVectorizer(BaseVectorizer):
    """
    Term Frequency (count) Vectorizer for documents and queries.
    Preserves exact vectorization math from ir_lib.
    """

    def __init__(self, processor: Optional[TextProcessor] = None):
        self.processor = processor if processor is not None else TextProcessor()

    def transform_documents(
        self,
        dataset: IRDataset,
        vocabulary: Vocabulary,
        pre_processing_mode: str = "none",
        text_column: Optional[str] = None
    ) -> List[np.ndarray]:
        col = text_column if text_column else dataset.text_column
        vocabulary_index = vocabulary.index_map
        vocab_len = len(vocabulary_index)
        document_vectors = []

        for document_text in dataset.df[col]:
            document_vector = np.zeros(vocab_len, dtype=int)
            tokens = self.processor.tokenize(str(document_text))
            tokens = self.processor.process_tokens(tokens, mode=pre_processing_mode)

            for token in tokens:
                if token in vocabulary_index:
                    token_index = vocabulary_index[token]
                    document_vector[token_index] += 1

            document_vectors.append(document_vector)

        return document_vectors

    def transform_queries(
        self,
        queries: Union[QueryDataset, List[Tuple[str, str]], List[str]],
        vocabulary: Vocabulary,
        pre_processing_mode: str = "none"
    ) -> List[np.ndarray]:
        vocabulary_index = vocabulary.index_map
        vocab_len = len(vocabulary_index)
        query_vectors = []

        # Convert queries to list of text strings
        query_texts: List[str] = []
        if isinstance(queries, QueryDataset):
            query_texts = [q.text for q in queries]
        elif isinstance(queries, list):
            for item in queries:
                if isinstance(item, tuple):
                    query_texts.append(item[0])
                elif isinstance(item, Query):
                    query_texts.append(item.text)
                else:
                    query_texts.append(str(item))

        for query_text in query_texts:
            query_vector = np.zeros(vocab_len, dtype=int)
            tokens = self.processor.tokenize(query_text)
            tokens = self.processor.process_tokens(tokens, mode=pre_processing_mode)

            for token in tokens:
                if token in vocabulary_index:
                    token_index = vocabulary_index[token]
                    query_vector[token_index] += 1

            query_vectors.append(query_vector)

        return query_vectors


class TFIDFVectorizer(BaseVectorizer):
    """
    TF-IDF Vectorizer for documents and queries.
    Preserves exact TF-IDF and IDF calculations:
      IDF = log(N / DF)
      TF = count / num_tokens
      TF-IDF = TF * IDF
    """

    def __init__(self, processor: Optional[TextProcessor] = None):
        self.processor = processor if processor is not None else TextProcessor()

    @staticmethod
    def get_document_frequency_and_tokenized_documents(
        dataset: IRDataset,
        vocabulary: Vocabulary,
        pre_processing_mode: str = "none",
        text_column: Optional[str] = None
    ) -> Tuple[np.ndarray, List[List[str]]]:
        col = text_column if text_column else dataset.text_column
        vocabulary_index = vocabulary.index_map
        processor = TextProcessor()
        document_frequency = np.zeros(len(vocabulary_index))
        tokenized_documents = []

        for document_text in dataset.df[col]:
            tokens = processor.tokenize(str(document_text))
            tokens = processor.process_tokens(tokens, mode=pre_processing_mode)
            tokenized_documents.append(tokens)

            unique_tokens = set(tokens)
            for token in unique_tokens:
                if token in vocabulary_index:
                    document_frequency[vocabulary_index[token]] += 1

        return document_frequency, tokenized_documents

    @staticmethod
    def get_idf(num_documents: int, document_frequency: np.ndarray) -> np.ndarray:
        return np.log(num_documents / document_frequency)

    @staticmethod
    def vectorize_tokens_tf_idf(
        tokens: List[str],
        vocabulary_index: Dict[str, int],
        idf: np.ndarray,
        vocabulary_size: int
    ) -> np.ndarray:
        tf_idf_vector = np.zeros(vocabulary_size)
        if not tokens:
            return tf_idf_vector

        token_counts = Counter(tokens)
        num_tokens = len(tokens)

        for word, count in token_counts.items():
            if word in vocabulary_index:
                tf = count / num_tokens
                word_index = vocabulary_index[word]
                tf_idf_vector[word_index] = tf * idf[word_index]

        return tf_idf_vector

    def transform_documents(
        self,
        dataset: IRDataset,
        vocabulary: Vocabulary,
        pre_processing_mode: str = "none",
        text_column: Optional[str] = None
    ) -> List[np.ndarray]:
        col = text_column if text_column else dataset.text_column
        num_documents = len(dataset)
        vocabulary_index = vocabulary.index_map
        vocabulary_size = len(vocabulary_index)

        document_frequency, tokenized_documents = self.get_document_frequency_and_tokenized_documents(
            dataset, vocabulary, pre_processing_mode=pre_processing_mode, text_column=col
        )
        idf = self.get_idf(num_documents, document_frequency)

        tf_idf_vectors = []
        for tokens in tokenized_documents:
            vec = self.vectorize_tokens_tf_idf(tokens, vocabulary_index, idf, vocabulary_size)
            tf_idf_vectors.append(vec)

        return tf_idf_vectors

    def transform_queries(
        self,
        queries: Union[QueryDataset, List[Tuple[str, str]], List[str]],
        dataset: IRDataset,
        vocabulary: Vocabulary,
        pre_processing_mode: str = "none",
        text_column: Optional[str] = None
    ) -> List[np.ndarray]:
        col = text_column if text_column else dataset.text_column
        num_documents = len(dataset)
        vocabulary_index = vocabulary.index_map
        vocabulary_size = len(vocabulary_index)

        document_frequency, _ = self.get_document_frequency_and_tokenized_documents(
            dataset, vocabulary, pre_processing_mode=pre_processing_mode, text_column=col
        )
        idf = self.get_idf(num_documents, document_frequency)

        query_texts: List[str] = []
        if isinstance(queries, QueryDataset):
            query_texts = [q.text for q in queries]
        elif isinstance(queries, list):
            for item in queries:
                if isinstance(item, tuple):
                    query_texts.append(item[0])
                elif isinstance(item, Query):
                    query_texts.append(item.text)
                else:
                    query_texts.append(str(item))

        tf_idf_query_vectors = []
        for query in query_texts:
            tokens = self.processor.tokenize(query)
            tokens = self.processor.process_tokens(tokens, mode=pre_processing_mode)
            vec = self.vectorize_tokens_tf_idf(tokens, vocabulary_index, idf, vocabulary_size)
            tf_idf_query_vectors.append(vec)

        return tf_idf_query_vectors

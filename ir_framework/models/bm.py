import math
import numpy as np
from typing import List, Tuple, Dict, Any, Optional
from ir_framework.models.base import BaseRetrievalModel
from ir_framework.data.dataset import IRDataset
from ir_framework.indexing.vocabulary import Vocabulary
from ir_framework.preprocessing.text_processor import TextProcessor


class BMContext:
    """
    Context structure for Okapi BM family models.
    Caches IDF, document frequencies, document sizes, and average document size.
    """

    def __init__(
        self,
        num_documents: int,
        document_frequency: np.ndarray,
        tokenized_documents: List[List[str]],
        document_sizes: List[int],
        average_document_size: float,
        idf: np.ndarray
    ):
        self.num_documents = num_documents
        self.document_frequency = document_frequency
        self.tokenized_documents = tokenized_documents
        self.document_sizes = document_sizes
        self.average_document_size = average_document_size
        self.idf = idf

    def to_dict(self) -> Dict[str, Any]:
        """Provides backward-compatible dictionary representation."""
        return {
            "num_documents": self.num_documents,
            "document_frequency": self.document_frequency,
            "tokenized_documents": self.tokenized_documents,
            "document_sizes": self.document_sizes,
            "average_document_size": self.average_document_size,
            "idf": self.idf
        }

    @classmethod
    def build_from_dataset(
        cls,
        dataset: IRDataset,
        vocabulary: Vocabulary,
        pre_processing_mode: str = "none",
        text_column: Optional[str] = None
    ) -> "BMContext":
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

        num_documents = len(dataset)
        size_accumulator = sum(len(tokens) for tokens in tokenized_documents)
        average_document_size = size_accumulator / num_documents if num_documents > 0 else 0.0
        idf = np.log(num_documents / document_frequency)
        document_sizes = [len(tokens) for tokens in tokenized_documents]

        return cls(
            num_documents=num_documents,
            document_frequency=document_frequency,
            tokenized_documents=tokenized_documents,
            document_sizes=document_sizes,
            average_document_size=average_document_size,
            idf=idf
        )


class BM1Model(BaseRetrievalModel):
    """BM1 Model ranking implementation."""

    def rank(
        self,
        query_vector: np.ndarray,
        dataset: IRDataset,
        num_documents_to_recover: int,
        bm_context: BMContext,
        vector_column: str = "document_vectors",
        **kwargs
    ) -> List[Tuple[int, float]]:
        dataset.load_vector_column(vector_column)
        ctx = bm_context.to_dict() if isinstance(bm_context, BMContext) else bm_context

        similarity_ranking = []
        num_documents = ctx["num_documents"]
        document_frequency = ctx["document_frequency"]
        query_term_indexes = np.where(query_vector != 0)[0]

        for document_index, document_vector in enumerate(dataset.df[vector_column]):
            similarity_score = 0.0
            for index in query_term_indexes:
                if document_vector[index] != 0:
                    ni = document_frequency[index]
                    similarity_score += math.log((num_documents - ni + 0.5) / (ni + 0.5))

            similarity_ranking.append((document_index, similarity_score))

        similarity_ranking.sort(key=lambda item: item[1], reverse=True)
        return similarity_ranking[:num_documents_to_recover]


class BM11Model(BaseRetrievalModel):
    """BM11 Model ranking implementation."""

    def __init__(self, k1: float = 1.0):
        self.k1 = k1

    def rank(
        self,
        query_vector: np.ndarray,
        dataset: IRDataset,
        num_documents_to_recover: int,
        bm_context: BMContext,
        k1: Optional[float] = None,
        vector_column: str = "document_vectors",
        **kwargs
    ) -> List[Tuple[int, float]]:
        dataset.load_vector_column(vector_column)
        ctx = bm_context.to_dict() if isinstance(bm_context, BMContext) else bm_context
        k1_val = k1 if k1 is not None else self.k1

        similarity_ranking = []
        idf = ctx["idf"]
        average_document_size = ctx["average_document_size"]
        document_sizes = ctx["document_sizes"]
        query_term_indexes = np.where(query_vector != 0)[0]

        for document_index, document_vector in enumerate(dataset.df[vector_column]):
            document_size = document_sizes[document_index]
            similarity_score = 0.0

            for index in query_term_indexes:
                term_frequency_in_document = document_vector[index]
                if term_frequency_in_document != 0:
                    similarity_score += idf[index] * (
                        (term_frequency_in_document * (k1_val + 1)) /
                        (term_frequency_in_document + (k1_val * (document_size / average_document_size)))
                    )

            similarity_ranking.append((document_index, similarity_score))

        similarity_ranking.sort(key=lambda item: item[1], reverse=True)
        return similarity_ranking[:num_documents_to_recover]


class BM15Model(BaseRetrievalModel):
    """BM15 Model ranking implementation."""

    def __init__(self, k1: float = 1.0):
        self.k1 = k1

    def rank(
        self,
        query_vector: np.ndarray,
        dataset: IRDataset,
        num_documents_to_recover: int,
        bm_context: BMContext,
        k1: Optional[float] = None,
        vector_column: str = "document_vectors",
        **kwargs
    ) -> List[Tuple[int, float]]:
        dataset.load_vector_column(vector_column)
        ctx = bm_context.to_dict() if isinstance(bm_context, BMContext) else bm_context
        k1_val = k1 if k1 is not None else self.k1

        similarity_ranking = []
        idf = ctx["idf"]
        query_term_indexes = np.where(query_vector != 0)[0]

        for document_index, document_vector in enumerate(dataset.df[vector_column]):
            similarity_score = 0.0

            for index in query_term_indexes:
                term_frequency_in_document = document_vector[index]
                if term_frequency_in_document != 0:
                    similarity_score += idf[index] * (
                        (term_frequency_in_document * (k1_val + 1)) /
                        (term_frequency_in_document + k1_val)
                    )

            similarity_ranking.append((document_index, similarity_score))

        similarity_ranking.sort(key=lambda item: item[1], reverse=True)
        return similarity_ranking[:num_documents_to_recover]


class BM25Model(BaseRetrievalModel):
    """BM25 Model ranking implementation."""

    def __init__(self, k1: float = 1.2, b: float = 0.75):
        self.k1 = k1
        self.b = b

    def rank(
        self,
        query_vector: np.ndarray,
        dataset: IRDataset,
        num_documents_to_recover: int,
        bm_context: BMContext,
        k1: Optional[float] = None,
        b: Optional[float] = None,
        vector_column: str = "document_vectors",
        **kwargs
    ) -> List[Tuple[int, float]]:
        dataset.load_vector_column(vector_column)
        ctx = bm_context.to_dict() if isinstance(bm_context, BMContext) else bm_context
        k1_val = k1 if k1 is not None else self.k1
        b_val = b if b is not None else self.b

        similarity_ranking = []
        idf = ctx["idf"]
        average_document_size = ctx["average_document_size"]
        document_sizes = ctx["document_sizes"]
        query_term_indexes = np.where(query_vector != 0)[0]

        for document_index, document_vector in enumerate(dataset.df[vector_column]):
            document_size = document_sizes[document_index]
            similarity_score = 0.0

            for index in query_term_indexes:
                term_frequency_in_document = document_vector[index]
                if term_frequency_in_document != 0:
                    similarity_score += idf[index] * (
                        (term_frequency_in_document * (k1_val + 1)) /
                        (term_frequency_in_document + (k1_val * (1 - b_val + b_val * (document_size / average_document_size))))
                    )

            similarity_ranking.append((document_index, similarity_score))

        similarity_ranking.sort(key=lambda item: item[1], reverse=True)
        return similarity_ranking[:num_documents_to_recover]

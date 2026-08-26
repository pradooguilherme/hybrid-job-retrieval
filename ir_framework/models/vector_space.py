import numpy as np
from typing import List, Tuple, Optional
from ir_framework.models.base import BaseRetrievalModel
from ir_framework.data.dataset import IRDataset


class VectorSpaceModel(BaseRetrievalModel):
    """
    Vector Space Model using Cosine Similarity for document ranking.
    Preserves exact mathematical calculations from ir_lib.
    """

    @staticmethod
    def compute_vector_norm(vector: np.ndarray) -> float:
        return float(np.linalg.norm(vector))

    @staticmethod
    def compute_cosine_similarity(
        query_vector: np.ndarray,
        document_vector: np.ndarray,
        query_norm: float,
        document_norm: float
    ) -> float:
        if query_norm == 0 or document_norm == 0:
            return 0.0
        return float(np.dot(query_vector, document_vector) / (query_norm * document_norm))

    @classmethod
    def get_document_norms(
        cls,
        dataset: IRDataset,
        vector_column: str = "document_vectors"
    ) -> List[float]:
        dataset.load_vector_column(vector_column)
        document_norms = []
        for document_vector in dataset.df[vector_column]:
            document_norms.append(cls.compute_vector_norm(document_vector))
        return document_norms

    def rank(
        self,
        query_vector: np.ndarray,
        dataset: IRDataset,
        num_documents_to_recover: int,
        document_norms: Optional[List[float]] = None,
        vector_column: str = "document_vectors",
        **kwargs
    ) -> List[Tuple[int, float]]:
        dataset.load_vector_column(vector_column)
        if document_norms is None:
            document_norms = self.get_document_norms(dataset, vector_column=vector_column)

        similarity_ranking = []
        query_norm = self.compute_vector_norm(query_vector)

        for document_index, document_vector in enumerate(dataset.df[vector_column]):
            similarity_score = self.compute_cosine_similarity(
                query_vector,
                document_vector,
                query_norm,
                document_norms[document_index]
            )
            similarity_ranking.append((document_index, similarity_score))

        similarity_ranking.sort(key=lambda item: item[1], reverse=True)
        return similarity_ranking[:num_documents_to_recover]

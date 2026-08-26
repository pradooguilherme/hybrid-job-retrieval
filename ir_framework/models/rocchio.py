import numpy as np
from typing import List, Dict, Any, Optional
from ir_framework.data.dataset import IRDataset
from ir_framework.models.vector_space import VectorSpaceModel
from ir_framework.evaluation.metrics import IREvaluator


class RocchioFeedback:
    """
    Implements Rocchio Relevance Feedback for query expansion/reweighting.
    Preserves exact formula:
      Q_new = alpha * Q_0 + (beta / |R|) * sum(R) - (gamma / |NR|) * sum(NR)
      Q_new = clip(Q_new, 0, None)
    """

    def __init__(self, alpha: float = 1.0, beta: float = 0.75, gamma: float = 0.15):
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

    @classmethod
    def compute_query(
        cls,
        query_vector: np.ndarray,
        relevant_vectors: List[np.ndarray],
        non_relevant_vectors: List[np.ndarray],
        alpha: float = 1.0,
        beta: float = 0.75,
        gamma: float = 0.15
    ) -> np.ndarray:
        original_component = alpha * query_vector

        if len(relevant_vectors) > 0:
            relevant_matrix = np.array(relevant_vectors)
            relevant_component = (beta / len(relevant_vectors)) * np.sum(relevant_matrix, axis=0)
        else:
            relevant_component = np.zeros_like(query_vector)

        if len(non_relevant_vectors) > 0:
            non_relevant_matrix = np.array(non_relevant_vectors)
            non_relevant_component = (gamma / len(non_relevant_vectors)) * np.sum(non_relevant_matrix, axis=0)
        else:
            non_relevant_component = np.zeros_like(query_vector)

        modified_query = original_component + relevant_component - non_relevant_component
        modified_query = np.clip(modified_query, a_min=0, a_max=None)
        return modified_query

    def update_query(
        self,
        query_vector: np.ndarray,
        relevant_vectors: List[np.ndarray],
        non_relevant_vectors: List[np.ndarray]
    ) -> np.ndarray:
        return self.compute_query(
            query_vector,
            relevant_vectors,
            non_relevant_vectors,
            alpha=self.alpha,
            beta=self.beta,
            gamma=self.gamma
        )

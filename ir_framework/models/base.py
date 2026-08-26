from abc import ABC, abstractmethod
from typing import List, Tuple
import numpy as np

from ir_framework.data.dataset import IRDataset


class BaseRetrievalModel(ABC):
    """Abstract Base Class for all IR Ranking / Retrieval Models."""

    @abstractmethod
    def rank(
        self,
        query_vector: np.ndarray,
        dataset: IRDataset,
        num_documents_to_recover: int,
        **kwargs
    ) -> List[Tuple[int, float]]:
        """
        Ranks documents in dataset for a given query vector.
        Returns a list of (document_index, similarity_score) sorted descending by score.
        """
        pass

import numpy as np
from typing import List, Tuple, Dict, Optional
from ir_framework.models.base import BaseRetrievalModel
from ir_framework.data.dataset import IRDataset


class HybridRetrievalModel(BaseRetrievalModel):
    """
    Hybrid Retrieval Model combining Lexical (e.g. BM25 / TF-IDF) and Semantic (Dense Embedding) rankings.
    Supports Reciprocal Rank Fusion (RRF) and Weighted Score Fusion.
    """

    def __init__(self, k_rrf: int = 60, alpha: float = 0.5):
        """
        :param k_rrf: Smoothing constant for Reciprocal Rank Fusion (standard default = 60).
        :param alpha: Weight given to lexical model in weighted fusion (0.0 to 1.0).
        """
        self.k_rrf = k_rrf
        self.alpha = alpha

    @staticmethod
    def reciprocal_rank_fusion(
        rankings_list: List[List[Tuple[int, float]]],
        k: int = 60
    ) -> List[Tuple[int, float]]:
        """
        Combines multiple ranked lists using Reciprocal Rank Fusion (RRF).
        RRF Score(d) = sum(1 / (k + rank_i(d)))
        """
        rrf_scores: Dict[int, float] = {}

        for ranked_list in rankings_list:
            for rank, (doc_id, _score) in enumerate(ranked_list, start=1):
                if doc_id not in rrf_scores:
                    rrf_scores[doc_id] = 0.0
                rrf_scores[doc_id] += 1.0 / (k + rank)

        # Sort descending by RRF score
        sorted_docs = sorted(rrf_scores.items(), key=lambda item: item[1], reverse=True)
        return sorted_docs

    @staticmethod
    def weighted_score_fusion(
        lexical_ranking: List[Tuple[int, float]],
        semantic_ranking: List[Tuple[int, float]],
        alpha: float = 0.5
    ) -> List[Tuple[int, float]]:
        """
        Combines two ranked lists using normalized weighted score fusion.
        Score(d) = alpha * norm(lexical_score(d)) + (1 - alpha) * norm(semantic_score(d))
        """
        lexical_dict = dict(lexical_ranking)
        semantic_dict = dict(semantic_ranking)
        all_docs = set(lexical_dict.keys()).union(set(semantic_dict.keys()))

        # Min-max normalization for lexical scores
        lex_vals = list(lexical_dict.values())
        min_lex, max_lex = (min(lex_vals), max(lex_vals)) if lex_vals else (0.0, 1.0)
        lex_range = max_lex - min_lex if max_lex > min_lex else 1.0

        # Min-max normalization for semantic scores
        sem_vals = list(semantic_dict.values())
        min_sem, max_sem = (min(sem_vals), max(sem_vals)) if sem_vals else (0.0, 1.0)
        sem_range = max_sem - min_sem if max_sem > min_sem else 1.0

        hybrid_scores: List[Tuple[int, float]] = []

        for doc_id in all_docs:
            lex_raw = lexical_dict.get(doc_id, min_lex)
            sem_raw = semantic_dict.get(doc_id, min_sem)

            lex_norm = (lex_raw - min_lex) / lex_range
            sem_norm = (sem_raw - min_sem) / sem_range

            combined_score = (alpha * lex_norm) + ((1.0 - alpha) * sem_norm)
            hybrid_scores.append((doc_id, combined_score))

        hybrid_scores.sort(key=lambda item: item[1], reverse=True)
        return hybrid_scores

    def rank(
        self,
        query_vector: np.ndarray,
        dataset: IRDataset,
        num_documents_to_recover: int,
        lexical_ranking: List[Tuple[int, float]] = None,
        semantic_ranking: List[Tuple[int, float]] = None,
        fusion_method: str = "rrf",
        **kwargs
    ) -> List[Tuple[int, float]]:
        if lexical_ranking is None or semantic_ranking is None:
            raise ValueError("HybridRetrievalModel requires both 'lexical_ranking' and 'semantic_ranking'.")

        if fusion_method == "rrf":
            combined_ranking = self.reciprocal_rank_fusion(
                [lexical_ranking, semantic_ranking],
                k=self.k_rrf
            )
        elif fusion_method == "weighted":
            combined_ranking = self.weighted_score_fusion(
                lexical_ranking,
                semantic_ranking,
                alpha=self.alpha
            )
        else:
            raise ValueError(f"Unknown fusion method '{fusion_method}'. Choose 'rrf' or 'weighted'.")

        return combined_ranking[:num_documents_to_recover]

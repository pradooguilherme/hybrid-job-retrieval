import math
import pandas as pd
from typing import List, Tuple, Union, Optional
from ir_framework.data.dataset import IRDataset


class IREvaluator:
    """
    Evaluation metrics calculator for IR systems.
    Computes Precision@K, Recall@K, Average Precision (AP), and NDCG@K.
    Dataset-agnostic: supports any IRDataset or DataFrame with configurable category columns.
    """

    @staticmethod
    def _extract_categories(
        dataset_or_df: Union[IRDataset, pd.DataFrame],
        category_column: Optional[str] = None
    ) -> Tuple[pd.DataFrame, str]:
        if isinstance(dataset_or_df, IRDataset):
            df = dataset_or_df.df
            col = category_column if category_column else dataset_or_df.category_column
        else:
            df = dataset_or_df
            col = category_column if category_column else "category"
        return df, col

    @classmethod
    def precision(
        cls,
        dataset_or_df: Union[IRDataset, pd.DataFrame],
        query_category: str,
        retrieved_documents: List[Tuple[int, float]],
        precision_nth: int,
        category_column: Optional[str] = None
    ) -> float:
        df, col = cls._extract_categories(dataset_or_df, category_column)
        num_relevant_documents_retrieval = 0

        for retrieved_document in retrieved_documents[:precision_nth]:
            document_category = df[col].iloc[retrieved_document[0]]
            if document_category == query_category:
                num_relevant_documents_retrieval += 1

        return num_relevant_documents_retrieval / precision_nth

    @classmethod
    def recall(
        cls,
        dataset_or_df: Union[IRDataset, pd.DataFrame],
        query_category: str,
        retrieved_documents: List[Tuple[int, float]],
        recall_nth: int,
        category_column: Optional[str] = None
    ) -> float:
        df, col = cls._extract_categories(dataset_or_df, category_column)
        total_rel = len(df[df[col] == query_category])
        if total_rel == 0:
            return 0.0

        num_rel = 0
        for retrieved_document in retrieved_documents[:recall_nth]:
            document_category = df[col].iloc[retrieved_document[0]]
            if document_category == query_category:
                num_rel += 1

        return num_rel / total_rel

    @classmethod
    def average_precision(
        cls,
        dataset_or_df: Union[IRDataset, pd.DataFrame],
        query_category: str,
        retrieved_documents: List[Tuple[int, float]],
        category_column: Optional[str] = None
    ) -> float:
        df, col = cls._extract_categories(dataset_or_df, category_column)
        total_rel = len(df[df[col] == query_category])
        if total_rel == 0:
            return 0.0

        num_rel = 0
        precision_sum = 0.0

        for i, retrieved_document in enumerate(retrieved_documents):
            document_category = df[col].iloc[retrieved_document[0]]
            if document_category == query_category:
                num_rel += 1
                precision_sum += num_rel / (i + 1)

        return precision_sum / total_rel

    @classmethod
    def ndcg(
        cls,
        dataset_or_df: Union[IRDataset, pd.DataFrame],
        query_category: str,
        retrieved_documents: List[Tuple[int, float]],
        ndcg_nth: int,
        category_column: Optional[str] = None
    ) -> float:
        df, col = cls._extract_categories(dataset_or_df, category_column)
        dcg = 0.0

        for i, retrieved_document in enumerate(retrieved_documents[:ndcg_nth]):
            document_category = df[col].iloc[retrieved_document[0]]
            if document_category == query_category:
                dcg += 1.0 / math.log(i + 2)

        total_rel = len(df[df[col] == query_category])
        ideal_rel = min(total_rel, ndcg_nth)

        idcg = 0.0
        for i in range(ideal_rel):
            idcg += 1.0 / math.log(i + 2)

        if idcg == 0.0:
            return 0.0

        return dcg / idcg

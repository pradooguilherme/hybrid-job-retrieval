import numpy as np
from typing import List, Union, Tuple, Optional
from ir_framework.data.dataset import IRDataset, QueryDataset, Query


class SemanticVectorizer:
    """
    Dense Semantic Vectorizer using SentenceTransformers (e.g. 'all-MiniLM-L6-v2').
    Encodes text documents and queries into dense continuous vector space embeddings.
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2", device: Optional[str] = None):
        self.model_name = model_name
        self.device = device
        self._model = None

    def _get_model(self):
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as e:
                raise ImportError(
                    "The 'sentence-transformers' package is required for SemanticVectorizer. "
                    "Please install it using: pip install sentence-transformers"
                ) from e
            self._model = SentenceTransformer(self.model_name, device=self.device)
        return self._model

    def transform_documents(
        self,
        dataset: IRDataset,
        text_column: Optional[str] = None,
        batch_size: int = 32,
        show_progress_bar: bool = True
    ) -> List[np.ndarray]:
        col = text_column if text_column else dataset.text_column
        model = self._get_model()
        texts = [str(text) for text in dataset.df[col]]
        embeddings = model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=show_progress_bar,
            convert_to_numpy=True,
            normalize_embeddings=True
        )
        return [vec for vec in embeddings]

    def transform_queries(
        self,
        queries: Union[QueryDataset, List[Tuple[str, str]], List[str], List[Query]],
        batch_size: int = 32,
        show_progress_bar: bool = False
    ) -> List[np.ndarray]:
        model = self._get_model()
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

        embeddings = model.encode(
            query_texts,
            batch_size=batch_size,
            show_progress_bar=show_progress_bar,
            convert_to_numpy=True,
            normalize_embeddings=True
        )
        return [vec for vec in embeddings]

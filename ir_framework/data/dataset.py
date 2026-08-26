import ast
import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Any, Optional, Union
from dataclasses import dataclass


def parse_vector(value: Any) -> np.ndarray:
    """Helper to convert stringified or list representations into numpy array."""
    if isinstance(value, np.ndarray):
        return value
    if isinstance(value, str):
        return np.array(ast.literal_eval(value))
    return np.array(value)


@dataclass
class Query:
    text: str
    category: str
    query_id: Optional[Any] = None


class QueryDataset:
    """Encapsulates a collection of queries for Information Retrieval."""

    def __init__(self, queries: Optional[List[Query]] = None):
        self.queries: List[Query] = queries if queries is not None else []

    @classmethod
    def from_tuples(cls, query_tuples: List[Tuple[str, str]]) -> "QueryDataset":
        """Build QueryDataset from a list of (text, category) tuples."""
        queries = [
            Query(text=text, category=cat, query_id=idx)
            for idx, (text, cat) in enumerate(query_tuples)
        ]
        return cls(queries)

    @classmethod
    def from_dataframe(
        cls,
        df: pd.DataFrame,
        text_column: str = "content",
        category_column: str = "category",
        id_column: Optional[str] = None
    ) -> "QueryDataset":
        """Build QueryDataset from a pandas DataFrame."""
        queries = []
        for idx, row in df.iterrows():
            q_id = row[id_column] if id_column and id_column in df.columns else idx
            queries.append(
                Query(
                    text=str(row[text_column]),
                    category=str(row[category_column]),
                    query_id=q_id
                )
            )
        return cls(queries)

    def __len__(self) -> int:
        return len(self.queries)

    def __getitem__(self, index: int) -> Query:
        return self.queries[index]

    def to_tuples(self) -> List[Tuple[str, str]]:
        return [(q.text, q.category) for q in self.queries]


class IRDataset:
    """
    Generic dataset container for Information Retrieval.
    Supports any IR dataset by specifying column names for document text, category, etc.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        text_column: str = "content",
        category_column: str = "category",
        id_column: Optional[str] = None
    ):
        self.df = df.copy()
        self.text_column = text_column
        self.category_column = category_column
        self.id_column = id_column

        if text_column not in self.df.columns:
            raise KeyError(f"Text column '{text_column}' not found in DataFrame.")

    @classmethod
    def from_csv(
        cls,
        path: str,
        delimiter: str = ",",
        text_column: str = "content",
        category_column: str = "category",
        id_column: Optional[str] = None
    ) -> "IRDataset":
        """Loads a dataset from a CSV file."""
        df = pd.read_csv(path, delimiter=delimiter)
        return cls(df, text_column=text_column, category_column=category_column, id_column=id_column)

    @classmethod
    def from_dataframe(
        cls,
        df: pd.DataFrame,
        text_column: str = "content",
        category_column: str = "category",
        id_column: Optional[str] = None
    ) -> "IRDataset":
        """Creates an IRDataset instance from a pandas DataFrame."""
        return cls(df, text_column=text_column, category_column=category_column, id_column=id_column)

    def __len__(self) -> int:
        return self.df.shape[0]

    def get_text(self, index: int) -> str:
        return str(self.df[self.text_column].iloc[index])

    def get_category(self, index: int) -> str:
        if self.category_column in self.df.columns:
            return str(self.df[self.category_column].iloc[index])
        return ""

    def get_all_texts(self) -> List[str]:
        return self.df[self.text_column].tolist()

    def get_all_categories(self) -> np.ndarray:
        if self.category_column in self.df.columns:
            return self.df[self.category_column].values
        return np.array([])

    def load_vector_column(self, column_name: str = "document_vectors") -> "IRDataset":
        """Ensures vector values in the column are loaded as numpy arrays."""
        target_col = column_name
        if target_col not in self.df.columns and target_col == "document_vectors" and "text_vectors" in self.df.columns:
            self.df = self.df.rename(columns={"text_vectors": "document_vectors"})
            target_col = "document_vectors"

        if target_col in self.df.columns and isinstance(self.df[target_col].iloc[0], str):
            self.df[target_col] = self.df[target_col].apply(parse_vector)

        return self

    def save_vector_column(
        self,
        document_vectors: List[np.ndarray],
        column_name: str = "document_vectors",
        output_path: Optional[str] = None
    ) -> None:
        """Stores document vectors into the dataset DataFrame and optionally saves to CSV."""
        self.df[column_name] = [v.tolist() for v in document_vectors]
        if output_path:
            self.df.to_csv(output_path, index=False)

    def get_random_document_vectors(
        self,
        num_documents: int,
        vector_column: str = "document_vectors"
    ) -> Dict[int, np.ndarray]:
        """Returns a mapping of random document indices to their vector representations."""
        self.load_vector_column(vector_column)
        random_vectors = {}
        total_docs = len(self)

        while len(random_vectors) < num_documents and len(random_vectors) < total_docs:
            doc_idx = np.random.randint(0, total_docs)
            if doc_idx not in random_vectors:
                random_vectors[doc_idx] = parse_vector(self.df[vector_column].iloc[doc_idx])

        return random_vectors

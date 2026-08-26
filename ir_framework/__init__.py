"""
Information Retrieval (IR) Framework
A modular, object-oriented framework for Information Retrieval tasks,
applicable to any IR dataset.
"""

from ir_framework.data.dataset import IRDataset, QueryDataset, Query
from ir_framework.preprocessing.text_processor import TextProcessor
from ir_framework.indexing.vocabulary import Vocabulary
from ir_framework.vectorization.vectorizers import CountVectorizer, TFIDFVectorizer
from ir_framework.vectorization.semantic_vectorizer import SemanticVectorizer
from ir_framework.models.vector_space import VectorSpaceModel
from ir_framework.models.bm import BM1Model, BM11Model, BM15Model, BM25Model, BMContext
from ir_framework.models.rocchio import RocchioFeedback
from ir_framework.models.hybrid import HybridRetrievalModel
from ir_framework.evaluation.metrics import IREvaluator
from ir_framework.pipeline.retrieval_pipeline import RetrievalPipeline

__all__ = [
    "IRDataset",
    "QueryDataset",
    "Query",
    "TextProcessor",
    "Vocabulary",
    "CountVectorizer",
    "TFIDFVectorizer",
    "SemanticVectorizer",
    "VectorSpaceModel",
    "BM1Model",
    "BM11Model",
    "BM15Model",
    "BM25Model",
    "BMContext",
    "HybridRetrievalModel",
    "RocchioFeedback",
    "IREvaluator",
    "RetrievalPipeline",
]

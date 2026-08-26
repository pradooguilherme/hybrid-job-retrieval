import os
import time
import pandas as pd
from typing import Dict, List, Tuple, Any, Optional, Union

from ir_framework.data.dataset import IRDataset
from ir_framework.evaluation.metrics import IREvaluator


class RetrievalPipeline:
    """
    Orchestrates retrieval processes, metric logging, output file writing,
    and performance measurement across models and datasets.
    """

    def __init__(self, dataset: IRDataset):
        self.dataset = dataset

    @staticmethod
    def build_metrics_dataframe(
        dataset_or_df: Union[IRDataset, pd.DataFrame],
        retrieval_results: Dict[int, List[Tuple[int, float]]],
        header_prefix: str,
        header_texts: Dict[int, Any],
        model_name: str,
        category_column: Optional[str] = None,
        text_column: Optional[str] = None
    ) -> pd.DataFrame:
        if isinstance(dataset_or_df, IRDataset):
            df = dataset_or_df.df
            cat_col = category_column if category_column else dataset_or_df.category_column
            txt_col = text_column if text_column else dataset_or_df.text_column
        else:
            df = dataset_or_df
            cat_col = category_column if category_column else "category"
            txt_col = text_column if text_column else "content"

        categories = df[cat_col].values
        rows = []

        for item_index, retrieved_documents in retrieval_results.items():
            if header_prefix == "Document":
                query_category = categories[item_index]
                query_text = df[txt_col].iloc[item_index]
            else:
                header_val = header_texts[item_index]
                if isinstance(header_val, tuple):
                    query_text, query_category = header_val
                else:
                    query_text = str(header_val)
                    query_category = categories[item_index] if item_index < len(categories) else ""

            rows.append({
                "model": model_name,
                "query_id": item_index,
                "query_text": query_text,
                "category": query_category,
                "P@10": IREvaluator.precision(df, query_category, retrieved_documents, 10, category_column=cat_col),
                "P@20": IREvaluator.precision(df, query_category, retrieved_documents, 20, category_column=cat_col),
                "P@30": IREvaluator.precision(df, query_category, retrieved_documents, 30, category_column=cat_col),
                "R@30": IREvaluator.recall(df, query_category, retrieved_documents, 30, category_column=cat_col),
                "NDCG@10": IREvaluator.ndcg(df, query_category, retrieved_documents, 10, category_column=cat_col),
                "NDCG@20": IREvaluator.ndcg(df, query_category, retrieved_documents, 20, category_column=cat_col),
                "NDCG@30": IREvaluator.ndcg(df, query_category, retrieved_documents, 30, category_column=cat_col),
                "AP": IREvaluator.average_precision(df, query_category, retrieved_documents, category_column=cat_col)
            })

        return pd.DataFrame(rows)

    @staticmethod
    def save_metrics(metrics_df: pd.DataFrame, path: str) -> None:
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        if os.path.exists(path):
            existing = pd.read_csv(path)
            metrics_df = pd.concat([existing, metrics_df], ignore_index=True)

        metrics_df.to_csv(path, index=False)

    @staticmethod
    def save_summary(metrics_df: pd.DataFrame, path: str) -> pd.DataFrame:
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        summary = (
            metrics_df
            .groupby("model")
            .agg({
                "vocab_size": "first",
                "exec_time_seconds": "first",
                "AP": "mean",
                "P@10": "mean",
                "P@20": "mean",
                "P@30": "mean",
                "R@30": "mean",
                "NDCG@10": "mean",
                "NDCG@20": "mean",
                "NDCG@30": "mean"
            })
            .rename(columns={"AP": "MAP"})
            .reset_index()
        )

        summary.to_csv(path, index=False)
        return summary

    def write_retrieval_results(
        self,
        output_index_path: str,
        output_text_path: str,
        header_prefix: str,
        header_texts: Dict[int, Any],
        retrieval_results: Dict[int, List[Tuple[int, float]]],
        model_name: str,
        category_column: Optional[str] = None,
        text_column: Optional[str] = None
    ) -> pd.DataFrame:
        os.makedirs(os.path.dirname(os.path.abspath(output_index_path)), exist_ok=True)
        os.makedirs(os.path.dirname(os.path.abspath(output_text_path)), exist_ok=True)

        df = self.dataset.df
        cat_col = category_column if category_column else self.dataset.category_column
        txt_col = text_column if text_column else self.dataset.text_column

        with open(output_index_path, "w", encoding="utf-8") as f_index, \
             open(output_text_path, "w", encoding="utf-8") as f_text:

            categories = df[cat_col].values
            ap_sum = 0.0
            num_queries = len(retrieval_results)

            for item_index, retrieved_documents in retrieval_results.items():

                if header_prefix == "Document":
                    query_category = categories[item_index]
                else:
                    header_val = header_texts[item_index]
                    if isinstance(header_val, tuple):
                        query_category = header_val[1]
                    else:
                        query_category = categories[item_index] if item_index < len(categories) else ""

                ap_q = IREvaluator.average_precision(df, query_category, retrieved_documents, category_column=cat_col)
                ap_sum += ap_q

                precision_10 = IREvaluator.precision(df, query_category, retrieved_documents, 10, category_column=cat_col)
                precision_20 = IREvaluator.precision(df, query_category, retrieved_documents, 20, category_column=cat_col)
                precision_30 = IREvaluator.precision(df, query_category, retrieved_documents, 30, category_column=cat_col)
                recall = IREvaluator.recall(df, query_category, retrieved_documents, 30, category_column=cat_col)
                ndcg10 = IREvaluator.ndcg(df, query_category, retrieved_documents, 10, category_column=cat_col)
                ndcg20 = IREvaluator.ndcg(df, query_category, retrieved_documents, 20, category_column=cat_col)
                ndcg30 = IREvaluator.ndcg(df, query_category, retrieved_documents, 30, category_column=cat_col)

                f_index.write(f"{header_prefix} {item_index}\n")
                f_index.write(f"Precision@10: {precision_10}\n")
                f_index.write(f"Precision@20: {precision_20}\n")
                f_index.write(f"Precision@30: {precision_30}\n")
                f_index.write(f"Recall@30: {recall}\n")
                f_index.write(f"NDCG@10: {ndcg10}\n")
                f_index.write(f"NDCG@20: {ndcg20}\n")
                f_index.write(f"NDCG@30: {ndcg30}\n\n")

                f_text.write(f"{header_prefix}: {header_texts[item_index]}\n")
                f_text.write(f"Precision@10: {precision_10}\n")
                f_text.write(f"Precision@20: {precision_20}\n")
                f_text.write(f"Precision@30: {precision_30}\n")
                f_text.write(f"Recall@30: {recall}\n")
                f_text.write(f"NDCG@10: {ndcg10}\n")
                f_text.write(f"NDCG@20: {ndcg20}\n")
                f_text.write(f"NDCG@30: {ndcg30}\n\n")

                for rank, retrieved_document in enumerate(retrieved_documents, start=1):
                    retrieved_document_index = retrieved_document[0]
                    retrieved_document_score = retrieved_document[1]

                    f_index.write(
                        f"Rank {rank} | Document Index: {retrieved_document_index} | Score: {retrieved_document_score}\n"
                    )

                    f_text.write(
                        f"Rank {rank} | Document Index: {retrieved_document_index} | Score: {retrieved_document_score} | Text: {df[txt_col].iloc[retrieved_document_index]}\n"
                    )

                f_index.write("\n")
                f_text.write("\n")

            map_metric = ap_sum / num_queries if num_queries > 0 else 0.0
            f_index.write(f"Map: {map_metric}\n")
            f_text.write(f"Map: {map_metric}\n")

        metrics_df = self.build_metrics_dataframe(
            df, retrieval_results, header_prefix, header_texts, model_name,
            category_column=cat_col, text_column=txt_col
        )
        return metrics_df

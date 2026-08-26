"""
Experimento de Recuperação Híbrida (Hybrid Search)
Combina:
  1. BM25 Otimizado (k1=3.0, b=0.85, com lemmatização) [Léxico]
  2. BAAI/bge-small-en-v1.5 [Semântico Denso]
Métodos de Fusão:
  - Reciprocal Rank Fusion (RRF, k=60)
  - Weighted Score Fusion (alpha=0.5)
"""

import os
import time
import pandas as pd
from ir_framework import (
    IRDataset,
    QueryDataset,
    Vocabulary,
    CountVectorizer,
    BM25Model,
    BMContext,
    SemanticVectorizer,
    VectorSpaceModel,
    HybridRetrievalModel,
    RetrievalPipeline,
)


def run_experiment():
    print("=== Configurando Experimento de Busca Híbrida (BM25 + BGE-Small) ===")

    # 1. Carregar Dataset de Documentos (Currículos)
    resume_path = "../dataset/cleaned/resume.csv"
    print(f"Carregando documentos de: {resume_path}")
    doc_dataset = IRDataset.from_csv(
        resume_path,
        text_column="resume",
        category_column="Category"
    )
    print(f"Total de currículos (documentos): {len(doc_dataset)}")

    # 2. Carregar Dataset de Consultas (Vagas)
    job_path = "../dataset/cleaned/job.csv"
    print(f"Carregando consultas de: {job_path}")
    job_df = pd.read_csv(job_path)
    job_df = job_df.groupby("category", group_keys=False).head(3).reset_index(drop=True)

    query_dataset = QueryDataset.from_dataframe(
        job_df,
        text_column="content",
        category_column="category"
    )
    print(f"Total de vagas selecionadas (3 por categoria): {len(query_dataset)}")

    start_time = time.perf_counter()

    # ---------------------------------------------------------
    # A. EXECUÇÃO DO MODELO LÉXICO (BM25 Otimizado)
    # ---------------------------------------------------------
    pre_processing_mode = "lemmatization"
    print(f"\n[1/3] Construindo vocabulário e vetorizando BM25 (modo: {pre_processing_mode})...")
    vocab = Vocabulary.from_dataset(doc_dataset, pre_processing_mode=pre_processing_mode)
    count_vectorizer = CountVectorizer()
    doc_vectors_bm = count_vectorizer.transform_documents(doc_dataset, vocab, pre_processing_mode=pre_processing_mode)
    doc_dataset.save_vector_column(doc_vectors_bm, column_name="count")
    query_vectors_bm = count_vectorizer.transform_queries(query_dataset, vocab, pre_processing_mode=pre_processing_mode)

    bm_context = BMContext.build_from_dataset(
        doc_dataset, vocab, pre_processing_mode=pre_processing_mode, text_column="resume"
    )
    bm25_model = BM25Model(k1=3.0, b=0.85)

    lexical_rankings = {}
    for query_idx in range(len(query_dataset)):
        query_vec = query_vectors_bm[query_idx]
        ranked_docs = bm25_model.rank(
            query_vector=query_vec,
            dataset=doc_dataset,
            num_documents_to_recover=100,
            bm_context=bm_context,
            vector_column="count"
        )
        lexical_rankings[query_idx] = ranked_docs

    # ---------------------------------------------------------
    # B. EXECUÇÃO DO MODELO SEMÂNTICO (BAAI/bge-small-en-v1.5)
    # ---------------------------------------------------------
    model_name = "BAAI/bge-small-en-v1.5"
    print(f"\n[2/3] Gerando embeddings semânticos com {model_name}...")
    semantic_vectorizer = SemanticVectorizer(model_name=model_name)
    doc_vectors_sem = semantic_vectorizer.transform_documents(doc_dataset, show_progress_bar=False)
    doc_dataset.save_vector_column(doc_vectors_sem, column_name="semantic_bge_small")
    query_vectors_sem = semantic_vectorizer.transform_queries(query_dataset, show_progress_bar=False)

    vsm_model = VectorSpaceModel()
    doc_norms_sem = vsm_model.get_document_norms(doc_dataset, vector_column="semantic_bge_small")

    semantic_rankings = {}
    for query_idx in range(len(query_dataset)):
        query_vec = query_vectors_sem[query_idx]
        ranked_docs = vsm_model.rank(
            query_vector=query_vec,
            dataset=doc_dataset,
            num_documents_to_recover=100,
            document_norms=doc_norms_sem,
            vector_column="semantic_bge_small"
        )
        semantic_rankings[query_idx] = ranked_docs

    # ---------------------------------------------------------
    # C. EXECUÇÃO DA FUSÃO HÍBRIDA (RRF & Weighted)
    # ---------------------------------------------------------
    print("\n[3/3] Executando fusão híbrida RRF (BM25 + BGE-Small)...")
    hybrid_model = HybridRetrievalModel(k_rrf=60, alpha=0.5)

    hybrid_rrf_results = {}
    hybrid_weighted_results = {}
    num_docs_to_recover = 30

    for query_idx in range(len(query_dataset)):
        # RRF Fusion
        ranked_rrf = hybrid_model.rank(
            query_vector=None,
            dataset=doc_dataset,
            num_documents_to_recover=num_docs_to_recover,
            lexical_ranking=lexical_rankings[query_idx],
            semantic_ranking=semantic_rankings[query_idx],
            fusion_method="rrf"
        )
        hybrid_rrf_results[query_idx] = ranked_rrf

        # Weighted Fusion
        ranked_weighted = hybrid_model.rank(
            query_vector=None,
            dataset=doc_dataset,
            num_documents_to_recover=num_docs_to_recover,
            lexical_ranking=lexical_rankings[query_idx],
            semantic_ranking=semantic_rankings[query_idx],
            fusion_method="weighted"
        )
        hybrid_weighted_results[query_idx] = ranked_weighted

    elapsed_time = time.perf_counter() - start_time

    # 4. Avaliar Métricas do RRF Híbrido
    print("\nAvaliando resultados do RRF Híbrido...")
    pipeline = RetrievalPipeline(doc_dataset)
    header_texts = {
        q_idx: (query_dataset[q_idx].text, query_dataset[q_idx].category)
        for q_idx in range(len(query_dataset))
    }

    metrics_df_rrf = pipeline.write_retrieval_results(
        output_index_path="../retrieval_results/hybrid_rrf_bge_bm25_index.txt",
        output_text_path="../retrieval_results/hybrid_rrf_bge_bm25_text.txt",
        header_prefix="JobQuery",
        header_texts=header_texts,
        retrieval_results=hybrid_rrf_results,
        model_name="Hybrid_RRF_BM25_BGE"
    )

    metrics_df_rrf["vocab_size"] = "Hybrid"
    metrics_df_rrf["exec_time_seconds"] = elapsed_time
    pipeline.save_metrics(metrics_df_rrf, "retrieval_results/hybrid_rrf_bge_bm25_metrics.csv")
    summary_rrf = pipeline.save_summary(metrics_df_rrf, "retrieval_results/hybrid_rrf_bge_bm25_summary.csv")

    print("\n=== RESUMO GERAL DAS MÉTRICAS HÍBRIDAS (RRF) ===")
    print(summary_rrf.to_string(index=False))

    # Avaliar Weighted Fusion Híbrido
    metrics_df_w = pipeline.write_retrieval_results(
        output_index_path="../retrieval_results/hybrid_weighted_bge_bm25_index.txt",
        output_text_path="../retrieval_results/hybrid_weighted_bge_bm25_text.txt",
        header_prefix="JobQuery",
        header_texts=header_texts,
        retrieval_results=hybrid_weighted_results,
        model_name="Hybrid_Weighted_BM25_BGE"
    )
    metrics_df_w["vocab_size"] = "Hybrid"
    metrics_df_w["exec_time_seconds"] = elapsed_time
    pipeline.save_metrics(metrics_df_w, "retrieval_results/hybrid_weighted_bge_bm25_metrics.csv")
    summary_w = pipeline.save_summary(metrics_df_w, "retrieval_results/hybrid_weighted_bge_bm25_summary.csv")

    print("\n=== RESUMO GERAL DAS MÉTRICAS HÍBRIDAS (WEIGHTED) ===")
    print(summary_w.to_string(index=False))


if __name__ == "__main__":
    run_experiment()

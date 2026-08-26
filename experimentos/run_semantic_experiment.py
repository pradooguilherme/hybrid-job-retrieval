"""
Experimento de Recuperação de Informação utilizando Busca Semântica (Dense Retrieval)
Modelo: SentenceTransformer ('all-MiniLM-L6-v2')
Documentos: Currículos (dataset/cleaned/resume.csv)
Consultas: Vagas de Emprego (dataset/cleaned/job.csv)
"""

import os
import time
import pandas as pd
from ir_framework import (
    IRDataset,
    QueryDataset,
    SemanticVectorizer,
    VectorSpaceModel,
    RetrievalPipeline,
)


def run_experiment():
    print("=== Configurando Experimento Semântico (all-MiniLM-L6-v2) ===")

    # 1. Carregar Dataset de Documentos (Currículos / Resumes)
    resume_path = "../dataset/cleaned/resume.csv"
    print(f"Carregando documentos de: {resume_path}")
    doc_dataset = IRDataset.from_csv(
        resume_path,
        text_column="resume",
        category_column="Category"
    )
    print(f"Total de currículos (documentos): {len(doc_dataset)}")

    # 2. Carregar Dataset de Consultas (Vagas / Jobs)
    job_path = "../dataset/cleaned/job.csv"
    print(f"Carregando consultas de: {job_path}")
    job_df = pd.read_csv(job_path)

    # Filtrar para utilizar no máximo 3 consultas por categoria
    job_df = job_df.groupby("category", group_keys=False).head(3).reset_index(drop=True)

    query_dataset = QueryDataset.from_dataframe(
        job_df,
        text_column="content",
        category_column="category"
    )
    print(f"Total de vagas selecionadas (3 por categoria): {len(query_dataset)}")

    start_time = time.perf_counter()

    # 3. Gerar Embeddings Densos dos Documentos e Consultas
    model_name = "all-MiniLM-L6-v2"
    print(f"\nVetorizando currículos com {model_name}...")
    semantic_vectorizer = SemanticVectorizer(model_name=model_name)
    doc_vectors = semantic_vectorizer.transform_documents(doc_dataset, show_progress_bar=True)
    doc_dataset.save_vector_column(doc_vectors, column_name="semantic_all_minilm")

    print(f"Vetorizando vagas com {model_name}...")
    query_vectors = semantic_vectorizer.transform_queries(query_dataset, show_progress_bar=False)

    # 4. Calcular similaridade de cosseno (os embeddings já estão normalizados L2)
    vsm_model = VectorSpaceModel()
    doc_norms = vsm_model.get_document_norms(doc_dataset, vector_column="semantic_all_minilm")

    # 5. Executar Recuperação (Ranking) para cada consulta
    print("\nExecutando recuperação semântica para cada vaga...")
    retrieval_results = {}
    num_docs_to_recover = 30

    for query_idx, query in enumerate(query_dataset):
        query_vector = query_vectors[query_idx]
        ranked_docs = vsm_model.rank(
            query_vector=query_vector,
            dataset=doc_dataset,
            num_documents_to_recover=num_docs_to_recover,
            document_norms=doc_norms,
            vector_column="semantic_all_minilm"
        )
        retrieval_results[query_idx] = ranked_docs

    elapsed_time = time.perf_counter() - start_time

    # 6. Avaliar Métricas usando o RetrievalPipeline
    print("\nAvaliando resultados semânticos...")
    pipeline = RetrievalPipeline(doc_dataset)

    header_texts = {
        q_idx: (query_dataset[q_idx].text, query_dataset[q_idx].category)
        for q_idx in range(len(query_dataset))
    }

    metrics_df = pipeline.write_retrieval_results(
        output_index_path="../retrieval_results/semantic_all_minilm_index.txt",
        output_text_path="../retrieval_results/semantic_all_minilm_text.txt",
        header_prefix="JobQuery",
        header_texts=header_texts,
        retrieval_results=retrieval_results,
        model_name="Semantic_all-MiniLM-L6-v2"
    )

    # Adicionar metadados
    metrics_df["vocab_size"] = 384  # Dimensão do vetor denso
    metrics_df["exec_time_seconds"] = elapsed_time
    pipeline.save_metrics(metrics_df, "retrieval_results/semantic_all_minilm_metrics.csv")
    summary = pipeline.save_summary(metrics_df, "retrieval_results/semantic_all_minilm_summary.csv")

    print("\n=== RESUMO GERAL DAS MÉTRICAS SEMÂNTICAS ===")
    print(summary.to_string(index=False))

    # Exemplo prático da primeira vaga
    print("\n=== EXEMPLO DE RECUPERAÇÃO DA PRIMEIRA VAGA ===")
    primeira_vaga = query_dataset[0]
    print(f"Vaga Categoria: {primeira_vaga.category}")
    print(f"Vaga Texto (primeiros 120 chars): {primeira_vaga.text[:120]}...")
    print("\nTop 5 Currículos Recuperados (Semântico):")
    for rank, (doc_idx, score) in enumerate(retrieval_results[0][:5], start=1):
        doc_cat = doc_dataset.get_category(doc_idx)
        is_relevant = "MATCH [V]" if doc_cat == primeira_vaga.category else "DIFF [X]"
        print(f"Rank {rank} | SimScore: {score:.4f} | Categoria: {doc_cat:<15} | {is_relevant}")


if __name__ == "__main__":
    run_experiment()

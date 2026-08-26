"""
Experimento de Recuperação de Informação utilizando BM25
Documentos: Currículos (dataset/cleaned/resume.csv)
Consultas: Vagas de Emprego (dataset/cleaned/job.csv)
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
    RetrievalPipeline,
)


def run_experiment():
    print("=== Configurando Experimento BM25 ===")

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

    # Filtrar para utilizar no máximo 3 consultas por categoria (conforme solicitado)
    job_df = job_df.groupby("category", group_keys=False).head(3).reset_index(drop=True)

    query_dataset = QueryDataset.from_dataframe(
        job_df,
        text_column="content",
        category_column="category"
    )
    print(f"Total de vagas selecionadas (3 por categoria): {len(query_dataset)}")

    # 3. Construir Vocabulário a partir do corpus de currículos com Lemmatization
    pre_processing_mode = "lemmatization"
    print(f"\nConstruindo vocabulário (modo: {pre_processing_mode})...")
    vocab = Vocabulary.from_dataset(doc_dataset, pre_processing_mode=pre_processing_mode)
    print(f"Tamanho do vocabulário: {len(vocab)} termos")

    start_time = time.perf_counter()

    # 4. Vetorizar Documentos e Consultas usando Frequência de Termos (CountVectorizer)
    print("\nVetorizando currículos com CountVectorizer...")
    count_vectorizer = CountVectorizer()
    doc_vectors = count_vectorizer.transform_documents(doc_dataset, vocab, pre_processing_mode=pre_processing_mode)
    doc_dataset.save_vector_column(doc_vectors, column_name="count")

    print("Vetorizando vagas com CountVectorizer...")
    query_vectors = count_vectorizer.transform_queries(query_dataset, vocab, pre_processing_mode=pre_processing_mode)

    # 5. Construir o contexto BMContext (calcula IDF, tamanhos de documentos e frequência de documentos)
    print("\nConstruindo contexto BMContext para cálculo do BM25...")
    bm_context = BMContext.build_from_dataset(
        doc_dataset,
        vocab,
        pre_processing_mode=pre_processing_mode,
        text_column="resume"
    )

    # 6. Instanciar o modelo BM25 com hiperparâmetros ajustados (k1=3.0, b=0.85)
    k1_opt, b_opt = 3.0, 0.85
    print(f"Instanciando BM25Model com k1={k1_opt}, b={b_opt}...")
    bm25_model = BM25Model(k1=k1_opt, b=b_opt)

    # 7. Executar Recuperação (Ranking) para cada consulta
    print("\nExecutando recuperação BM25 dos currículos mais relevantes para cada vaga...")
    retrieval_results = {}
    num_docs_to_recover = 30

    for query_idx, query in enumerate(query_dataset):
        query_vector = query_vectors[query_idx]
        ranked_docs = bm25_model.rank(
            query_vector=query_vector,
            dataset=doc_dataset,
            num_documents_to_recover=num_docs_to_recover,
            bm_context=bm_context,
            vector_column="count"
        )
        retrieval_results[query_idx] = ranked_docs

    elapsed_time = time.perf_counter() - start_time

    # 8. Avaliar Métricas usando o RetrievalPipeline
    print("\nAvaliando resultados...")
    pipeline = RetrievalPipeline(doc_dataset)

    # Montar mapeamento de textos e categorias das consultas
    header_texts = {
        q_idx: (query_dataset[q_idx].text, query_dataset[q_idx].category)
        for q_idx in range(len(query_dataset))
    }

    metrics_df = pipeline.write_retrieval_results(
        output_index_path="../retrieval_results/bm25_job_resume_index.txt",
        output_text_path="../retrieval_results/bm25_job_resume_text.txt",
        header_prefix="JobQuery",
        header_texts=header_texts,
        retrieval_results=retrieval_results,
        model_name="BM25_Job_Resume"
    )

    # Salvar métricas agregadas por categoria e médias gerais
    metrics_df["vocab_size"] = len(vocab)
    metrics_df["exec_time_seconds"] = elapsed_time
    pipeline.save_metrics(metrics_df, "retrieval_results/bm25_job_resume_metrics.csv")
    summary = pipeline.save_summary(metrics_df, "retrieval_results/bm25_job_resume_summary.csv")

    print("\n=== RESUMO GERAL DAS MÉTRICAS ===")
    print(summary.to_string(index=False))

    # Exibir um exemplo prático dos 5 primeiros currículos recuperados para a primeira vaga
    print("\n=== EXEMPLO DE RECUPERAÇÃO DA PRIMEIRA VAGA ===")
    primeira_vaga = query_dataset[0]
    print(f"Vaga Categoria: {primeira_vaga.category}")
    print(f"Vaga Texto (primeiros 120 chars): {primeira_vaga.text[:120]}...")
    print("\nTop 5 Currículos Recuperados:")
    for rank, (doc_idx, score) in enumerate(retrieval_results[0][:5], start=1):
        doc_cat = doc_dataset.get_category(doc_idx)
        is_relevant = "MATCH [V]" if doc_cat == primeira_vaga.category else "DIFF [X]"
        print(f"Rank {rank} | BM25Score: {score:.4f} | Categoria: {doc_cat:<15} | {is_relevant}")


if __name__ == "__main__":
    run_experiment()

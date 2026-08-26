"""
Experimento de Recuperação de Informação utilizando TF-IDF
Documentos: Currículos (dataset/cleaned/resume.csv)
Consultas: Vagas de Emprego (dataset/cleaned/job.csv)
"""

import os
import pandas as pd
from ir_framework import (
    IRDataset,
    QueryDataset,
    Vocabulary,
    TFIDFVectorizer,
    VectorSpaceModel,
    IREvaluator,
    RetrievalPipeline,
)


def run_experiment():
    print("=== Configurando Experimento TF-IDF ===")

    # 1. Carregar Dataset de Documentos (Currículos / Resumes)
    # Note que no resume.csv a coluna de texto é 'resume' e a de categoria é 'Category'
    resume_path = "../dataset/cleaned/resume.csv"
    print(f"Carregando documentos de: {resume_path}")
    doc_dataset = IRDataset.from_csv(
        resume_path,
        text_column="resume",
        category_column="Category"
    )
    print(f"Total de currículos (documentos): {len(doc_dataset)}")

    # 2. Carregar Dataset de Consultas (Vagas / Jobs)
    # No job.csv a coluna de texto é 'content' e a de categoria é 'category'
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

    # 3. Construir Vocabulário a partir do corpus de currículos (modo: lemmatization)
    pre_processing_mode = "lemmatization"
    print(f"\nConstruindo vocabulário (modo: {pre_processing_mode})...")
    vocab = Vocabulary.from_dataset(doc_dataset, pre_processing_mode=pre_processing_mode)
    print(f"Tamanho do vocabulário: {len(vocab)} termos")

    import time
    start_time = time.perf_counter()

    # 4. Vetorizar Documentos e Consultas usando TF-IDF
    print("\nVetorizando currículos com TF-IDF...")
    tfidf_vectorizer = TFIDFVectorizer()
    doc_vectors = tfidf_vectorizer.transform_documents(doc_dataset, vocab, pre_processing_mode=pre_processing_mode)
    doc_dataset.save_vector_column(doc_vectors, column_name="tf_idf")

    print("Vetorizando vagas com TF-IDF...")
    query_vectors = tfidf_vectorizer.transform_queries(query_dataset, doc_dataset, vocab, pre_processing_mode=pre_processing_mode)

    # 5. Pré-calcular normas dos documentos para otimização da Similaridade de Cosseno
    vsm_model = VectorSpaceModel()
    doc_norms = vsm_model.get_document_norms(doc_dataset, vector_column="tf_idf")

    # 6. Executar Recuperação (Ranking) para cada consulta
    print("\nExecutando recuperação dos currículos mais relevantes para cada vaga...")
    retrieval_results = {}
    num_docs_to_recover = 30

    for query_idx, query in enumerate(query_dataset):
        query_vector = query_vectors[query_idx]
        ranked_docs = vsm_model.rank(
            query_vector=query_vector,
            dataset=doc_dataset,
            num_documents_to_recover=num_docs_to_recover,
            document_norms=doc_norms,
            vector_column="tf_idf"
        )
        retrieval_results[query_idx] = ranked_docs

    elapsed_time = time.perf_counter() - start_time

    # 7. Avaliar Métricas usando o RetrievalPipeline e IREvaluator
    print("\nAvaliando resultados...")
    pipeline = RetrievalPipeline(doc_dataset)

    # Montar mapeamento de textos e categorias das consultas
    header_texts = {
        q_idx: (query_dataset[q_idx].text, query_dataset[q_idx].category)
        for q_idx in range(len(query_dataset))
    }

    metrics_df = pipeline.write_retrieval_results(
        output_index_path="../retrieval_results/tfidf_job_resume_index.txt",
        output_text_path="../retrieval_results/tfidf_job_resume_text.txt",
        header_prefix="JobQuery",
        header_texts=header_texts,
        retrieval_results=retrieval_results,
        model_name="TF-IDF_Job_Resume"
    )

    # Salvar métricas agregadas por categoria e médias gerais
    metrics_df["vocab_size"] = len(vocab)
    metrics_df["exec_time_seconds"] = elapsed_time
    pipeline.save_metrics(metrics_df, "retrieval_results/tfidf_job_resume_metrics.csv")
    summary = pipeline.save_summary(metrics_df, "retrieval_results/tfidf_job_resume_summary.csv")

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
        print(f"Rank {rank} | SimScore: {score:.4f} | Categoria: {doc_cat:<15} | {is_relevant}")


if __name__ == "__main__":
    run_experiment()

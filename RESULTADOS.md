# Documentação de Resultados e Hipóteses de Recuperação de Informação (RI)
**Projeto:** Job Matching (Vagas vs. Currículos)

---

## 1. Resumo Executivo
Este documento detalha o desenvolvimento, validação de hipóteses e avaliação de desempenho do sistema de Recuperação de Informação (RI) para matching entre vagas de emprego (*queries*) e currículos (*documentos*).

Foram avaliados e comparados os seguintes modelos em 3 paradigmas de RI:
1. **BM25 Baseline** ($k_1 = 1.2$, $b = 0.75$, sem lemmatização) — *Sintático/Probabilístico*
2. **BM25 Otimizado** ($k_1 = 3.0$, $b = 0.85$, com lemmatização NLTK WordNet) — *Sintático/Probabilístico Otimizado*
3. **TF-IDF Baseline** (VSM Cosseno, sem lemmatização) — *Sintático/Espaço Vetorial Espasso*
4. **TF-IDF Otimizado** (VSM Cosseno, com lemmatização) — *Sintático/Espaço Vetorial Espasso Otimizado*
5. **Semantic Retrieval (`all-MiniLM-L6-v2`)** — *Semântico / Dense Retrieval (Sentence Transformers)*
6. **Semantic Retrieval (`BAAI/bge-small-en-v1.5`)** — *Semântico / Dense Retrieval (SOTA Small Model)*
7. **Hybrid Search (BM25 + BGE-Small via RRF)** — *Busca Híbrida (Reciprocal Rank Fusion, $k=60$)*
8. **Hybrid Search (BM25 + BGE-Small via Weighted Fusion)** — *Busca Híbrida (Weighted Score Fusion, $\alpha=0.5$)*

---

## 2. Hipóteses Levantadas e Validação Experimental

### 💡 Hipótese 1: Impacto da Lemmatização no Vocabulário e Ruído Léxico
- **Hipótese:** O uso da lemmatização (`WordNetLemmatizer`) converte termos flexionados para suas formas léxicas fundamentais (*lemmas*), reduzindo a esparsidade do vocabulário.
- **Validação:** A lemmatização reduziu o vocabulário de **29.938** para **26.520** termos únicos (compressão de **~11,4%**). No BM25 o MAP subiu **+52,7%** e no TF-IDF a P@10 subiu **+5,79%**.

### 💡 Hipótese 2: Saturação da Frequência de Termos ($k_1$) em Domínio Profissional
- **Hipótese:** Em descrições de vagas e currículos, habilidades fundamentais reaparecem em várias seções. O parâmetro $k_1 = 3.0$ recompensa currículos que reforçam competências essenciais ao longo do documento.
- **Validação:** O ajuste $k_1 = 3.0$ gerou acréscimo relevante de precisão na busca BM25.

### 💡 Hipótese 3: Superioridade dos Modelos Semânticos Densos
- **Hipótese:** Modelos de embeddings densos (`BAAI/bge-small-en-v1.5`) superam os limites do *vocabulary mismatch*, capturando intenção e contexto profissional.
- **Validação:** O modelo `BAAI/bge-small-en-v1.5` alcançou a maior precisão individual (**P@10 = 75,56%** e **MAP = 0,1576**).

### 💡 Hipótese 4: Robustez da Busca Híbrida em Categorias Conflitantes (RRF vs Modelo Puro)
- **Hipótese:** Modelos puramente semânticos diluem siglas exatas e especificidades sintáticas (ex: certificações *SHRM*, *HRIS*, *Payroll* em RH). A Busca Híbrida via **Reciprocal Rank Fusion (RRF)** combina a sensibilidade léxica com a profundidade semântica do BGE-Small, eliminando gargalos de categorias específicas.
- **Validação:** Na categoria **HR**, o modelo semântico puro teve P@10 de apenas `33,33%`. A **Busca Híbrida RRF elevou a precisão para 56,67%**, superando tanto o modelo semântico puro quanto o BM25 puro ($53,33\%$). Além disso, no RRF a precisão nas categorias `ACCOUNTANT` e `DESIGNER` atingiu **100,0%**.

### 💡 Hipótese 5: Superioridade da Fusão do Melhor Modelo Léxico (TF-IDF) com o Melhor Semântico (BGE-Small)
- **Hipótese:** Como o **TF-IDF Otimizado** obteve o maior MAP ($0,1061$) e P@10 ($55,83\%$) entre os modelos estatísticos sintáticos, sua combinação com o **BGE-Small** via RRF deve superar a fusão BM25 + BGE em consistência e taxa de acerto.
- **Validação:** Confirmado. O **Hybrid RRF (TF-IDF + BGE-Small)** alcançou **MAP de 0,1501** (+15,3% vs RRF BM25+BGE) e **P@10 de 71,67%** (+1,11% vs RRF BM25+BGE). Na categoria `ENGINEERING`, a precisão saltou de $43,33\%$ (RRF BM25+BGE) para **$56,67\%$**.

---

## 3. Tabela Comparativa Completa dos Experimentos

| Métrica | BM25 Otimizado | TF-IDF Otimizado | Semantic (`bge-small-en-v1.5`) | Hybrid BM25+BGE (RRF $k=60$) | **Hybrid TF-IDF+BGE (Weighted $\alpha=0.5$)** | **Hybrid TF-IDF+BGE (RRF $k=60$)** |
|---|---|---|---|---|---|---|
| **Tipo de Modelo** | Sintático | Sintático | Semântico SOTA | Híbrido (BM25+BGE) | Híbrido (TFIDF+BGE) | **Híbrido (TFIDF+BGE RRF)** |
| **Dimensão / Vocab** | 26.520 | 26.520 | 384 | Híbrido | 26.520 / 384 | **26.520 / 384** |
| **Tempo Execução** | **7,15 s** | 131,54 s | 448,85 s | 355,20 s | 506,39 s | **506,39 s** |
| **MAP** | 0,0556 | 0,1061 | **0,1576** | 0,1302 | 0,1422 | **0,1501 (+15.3% vs RRF BM25)** |
| **P@10** | 0,4278 | 0,5583 | **0,7556** | 0,7056 | 0,6639 | **0,7167 (+28.4% vs TF-IDF)** |
| **P@20** | 0,3903 | 0,5417 | **0,7056** | 0,6472 | 0,6583 | **0,6903** |
| **P@30** | 0,3528 | 0,5222 | **0,6861** | 0,6019 | 0,6491 | **0,6574** |
| **Recall@30** | 0,0939 | 0,1392 | **0,1821** | 0,1600 | 0,1727 | **0,1750** |
| **NDCG@10** | 0,4215 | 0,5493 | **0,7455** | 0,7112 | 0,6796 | **0,7309** |

---

## 4. Análise de Desempenho por Categoria de Vaga (Léxico vs Semântico vs Híbrido)

| Categoria Profissional | BM25 Otimizado | TF-IDF Otimizado | Semantic (`bge-small-en-v1.5`) | Hybrid RRF (BM25 + BGE) | **Hybrid RRF (TF-IDF + BGE)** | Análise do Comportamento Híbrido |
|---|---|---|---|---|---|---|
| **ACCOUNTANT** | 0,6667 | 0,9000 | **0,9667** | 1,0000 | **0,9333** | Mantém altíssima precisão no Top 10. |
| **ADVOCATE** | 0,3333 | 0,2667 | **0,8667** | 0,7333 | **0,7000** | Forte salto semântico vs modelos léxicos puros. |
| **BANKING** | 0,3333 | 0,4333 | 0,5667 | 0,6000 | **0,5667** | Mantém desempenho sólido do semântico. |
| **CONSTRUCTION** | 0,5667 | 1,0000 | **1,0000** | 0,8667 | **1,0000** | **Alcançou 100% de precisão**. |
| **CONSULTANT** | 0,2000 | 0,1500 | **0,8000** | 0,6000 | **0,6000** | Mantém 4x mais precisão que o TF-IDF puro. |
| **DESIGNER** | 0,8000 | 0,7667 | **1,0000** | 1,0000 | **1,0000** | **100% de precisão mantida no Top 10**. |
| **ENGINEERING** | 0,2333 | 0,2500 | 0,5000 | 0,4333 | **0,5667** | **Supera todos os modelos anteriores (+30.8% vs RRF BM25)**. |
| **FINANCE** | 0,4333 | 0,6833 | **0,9333** | 0,8667 | **0,8667** | Consistência no Top 10. |
| **HEALTHCARE** | 0,5333 | 0,6167 | **0,7333** | 0,6667 | **0,7000** | Equilíbrio perfeito entre termos e contexto. |
| **HR** | 0,5333 | 0,6667 | 0,3333 | 0,5667 | **0,5000** | **Supera o gargalo semântico do BGE puro (33.3% -> 50%)**. |
| **SALES** | 0,0000 | 0,1667 | **0,4333** | 0,2667 | **0,2333** | Supera o modelo TF-IDF puro. |
| **TEACHER** | 0,5000 | 0,6167 | **0,9333** | 0,8667 | **0,9333** | **Empata com a precisão máxima do BGE puro**. |

---

## 5. Modificações Realizadas na Framework (`ir_framework`)

1. [`ir_framework/models/hybrid.py`](file:///home/pradooguilherme/Documents/Projetos/projeto-final-RI/ir_framework/models/hybrid.py):
   - Criada a classe `HybridRetrievalModel` com algoritmos de fusão **RRF (Reciprocal Rank Fusion)** e **Weighted Score Fusion**.
2. [`experimentos/run_hybrid_experiment.py`](file:///home/pradooguilherme/Documents/Projetos/projeto-final-RI/experimentos/run_hybrid_experiment.py):
   - Script de execução do experimento híbrido unindo BM25 Otimizado + BAAI/bge-small-en-v1.5.
3. [`experimentos/run_hybrid_tfidf_bge_experiment.py`](file:///home/pradooguilherme/Documents/Projetos/projeto-final-RI/experimentos/run_hybrid_tfidf_bge_experiment.py):
   - Script de execução do experimento híbrido unindo **TF-IDF Otimizado (com lematização)** + **BAAI/bge-small-en-v1.5**.

---

## 6. Arquivos de Saída dos Experimentos

- [`retrieval_results/hybrid_rrf_bge_tfidf_summary.csv`](file:///home/pradooguilherme/Documents/Projetos/projeto-final-RI/retrieval_results/hybrid_rrf_bge_tfidf_summary.csv)
- [`retrieval_results/hybrid_rrf_bge_tfidf_metrics.csv`](file:///home/pradooguilherme/Documents/Projetos/projeto-final-RI/retrieval_results/hybrid_rrf_bge_tfidf_metrics.csv)
- [`retrieval_results/hybrid_weighted_bge_tfidf_summary.csv`](file:///home/pradooguilherme/Documents/Projetos/projeto-final-RI/retrieval_results/hybrid_weighted_bge_tfidf_summary.csv)
- [`retrieval_results/hybrid_rrf_bge_bm25_summary.csv`](file:///home/pradooguilherme/Documents/Projetos/projeto-final-RI/retrieval_results/hybrid_rrf_bge_bm25_summary.csv)
- [`retrieval_results/hybrid_rrf_bge_bm25_metrics.csv`](file:///home/pradooguilherme/Documents/Projetos/projeto-final-RI/retrieval_results/hybrid_rrf_bge_bm25_metrics.csv)
- [`retrieval_results/semantic_bge_small_summary.csv`](file:///home/pradooguilherme/Documents/Projetos/projeto-final-RI/retrieval_results/semantic_bge_small_summary.csv)
- [`retrieval_results/tfidf_job_resume_summary.csv`](file:///home/pradooguilherme/Documents/Projetos/projeto-final-RI/retrieval_results/tfidf_job_resume_summary.csv)
- [`retrieval_results/bm25_job_resume_summary.csv`](file:///home/pradooguilherme/Documents/Projetos/projeto-final-RI/retrieval_results/bm25_job_resume_summary.csv)


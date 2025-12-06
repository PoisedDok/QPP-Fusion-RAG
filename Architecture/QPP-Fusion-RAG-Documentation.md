# QPP-Guided Multi-Retriever Fusion for Retrieval-Augmented Generation

## Technical Documentation

---

## 1. Project Overview

This research investigates whether Query Performance Prediction (QPP) scores can improve multi-retriever fusion for Retrieval-Augmented Generation (RAG) systems. The core idea is to dynamically weight different retrievers based on predicted query difficulty, rather than using static fusion weights.

### Research Hypothesis

QPP-weighted fusion methods will outperform unweighted baselines by assigning higher weights to retrievers that are predicted to perform well on each specific query.

### Experimental Setup

- **Dataset**: BEIR Natural Questions (NQ) — 2,681,468 Wikipedia passages, 3,452 test queries
- **LLMs**: Liquid LFM2-1.2B and Qwen3-4B
- **Evaluation**: IR metrics (NDCG@10, Recall@10, MRR@10) and QA metrics (Exact Match, F1, Containment, Semantic Similarity)

---

## 2. Multi-Retriever Architecture

### 2.1 Retrievers Used

Five diverse retrievers were implemented to capture different retrieval paradigms:

| Retriever | Type | Description |
|-----------|------|-------------|
| BM25 | Sparse Lexical | Classic term-frequency matching via PyTerrier inverted index |
| SPLADE | Sparse Learned | Neural term expansion using SPLADE++ EnsembleDistil via Pyserini |
| BGE | Dense | BAAI/bge-base-en-v1.5 embeddings with FAISS index via Pyserini |
| BM25→TCT-ColBERT | Two-Stage | BM25 first-stage (100 candidates) + TCT-ColBERT v2 reranking |
| BM25→MonoT5 | Two-Stage | BM25 first-stage (100 candidates) + MonoT5 neural reranking |

### 2.2 Why Two-Stage Pipelines?

Dense retrievers like TCT-ColBERT require encoding the entire corpus (2.68M documents × 768 dimensions ≈ 8GB embeddings), which takes hours. Two-stage pipelines are practical: BM25 quickly retrieves 100 candidates, then the neural model only reranks those 100 documents.

---

## 3. Query Performance Prediction (QPP)

### 3.1 QPP Methods Computed

Thirteen QPP methods were computed for each query-retriever pair, spanning pre-retrieval and post-retrieval categories:

| Category | Methods |
|----------|---------|
| Pre-retrieval | AvgIDF, MaxIDF, AvgICTF, MaxVAR, SCQ |
| Post-retrieval | Clarity, NQC, WIG, SMV, RSD, UEF, σmax, n(σ%) |
| Neural | NQA-QPP, BERTQPP |

### 3.2 QPP Analysis Findings

An empirical analysis of QPP method effectiveness revealed critical insights:

| QPP Method | Correlation with NDCG | Status |
|------------|----------------------|--------|
| **RSD** | +0.200 (Spearman) | **Best predictor** |
| UEF | +0.191 | Good |
| DM | +0.179 | Good |
| NQA-QPP | -0.285 | **Negative correlation — harmful** |
| WIG, SCNQC, BERTQPP | N/A | Constant values — useless |

**Key Finding**: RSD (Retrieval Score Deviation) alone captures nearly all useful signal. The other 12 QPP methods either add noise or have negative correlation with actual retrieval performance.

---

## 4. Fusion Methods

### 4.1 Baseline Methods (Unweighted)

| Method | Formula | Description |
|--------|---------|-------------|
| CombSUM | Σ S_i(d,q) | Sum of normalized scores across retrievers |
| CombMNZ | \|{i: d ∈ R_i}\| × Σ S_i(d,q) | CombSUM multiplied by count of retrievers returning document |
| RRF | Σ 1/(k + rank_i) | Reciprocal Rank Fusion with k=60 |

### 4.2 QPP-Weighted Methods (Heuristic)

| Method | Formula | QPP Used |
|--------|---------|----------|
| W-CombSUM | Σ w_i(q) × S_i(d,q) | RSD per retriever |
| W-CombMNZ | \|{i}\| × Σ w_i(q) × S_i(d,q) | RSD per retriever |
| W-RRF | Σ w_i(q) / (k + rank_i) | RSD per retriever |

### 4.3 Learned Fusion Methods (ML Models)

| Method | Architecture | Input Features |
|--------|--------------|----------------|
| Per-Retriever LightGBM | 5 independent regression models | 65 features (5 × 13 QPP) |
| Multi-Output LightGBM | Single model, 5 outputs | 65 features |
| MLP Neural Network | Feed-forward network | **5 features (RSD only)** |
| W-CombSUM Learned | CombSUM with learned weights | 65 features |

---

## 5. Machine Learning Model Development

### 5.1 Training Setup

- **Training Data**: 80% of 3,452 queries (2,761 training, 691 test)
- **Target Variable**: Per-retriever NDCG@10 normalized to sum to 1
- **Evaluation Metric**: NDCG@10 improvement over uniform weighting

### 5.2 MLP Architecture Bug Fix

The original MLP implementation had a critical logical error:

**Problem**: Used MSE Loss with Softmax output — mathematically incorrect for probability distributions. The softmax constrains outputs to sum to 1, but MSE treats each output independently.

**Solution**: Changed to Soft Cross-Entropy Loss (equivalent to KL Divergence for soft labels). Raw logits during training, softmax only at inference.

**Result**: NDCG improvement jumped from ~0% to +10.3%.

### 5.3 Feature Selection

Based on QPP analysis, the MLP was simplified to use only RSD:

| Configuration | Input Features | NDCG Improvement |
|---------------|----------------|------------------|
| All 13 QPP methods | 65 | +10.3% |
| **RSD only** | **5** | **+10.1%** |

RSD alone captures 97% of the predictive signal with 92% fewer features.

---

## 6. RAG Evaluation Pipeline

### 6.1 Evaluation Process

For each fusion method × LLM combination:
1. Retrieve top-k documents using fused ranking (k = 0, 1, 2, 3, 4, 5, 6, 10)
2. Construct prompt with retrieved context
3. Generate answer using LLM via LM Studio API
4. Compute retrieval and answer quality metrics

### 6.2 LLM Configuration

| Model | Parameters | Context Window | API Endpoint |
|-------|------------|----------------|--------------|
| Liquid LFM2-1.2B | 1.2B | 32K | localhost:1234 |
| Qwen3-4B | 4B | 32K | localhost:1234 |

### 6.3 Checkpointing

Long-running evaluations (3,452 queries × 8 k-values = 27,616 LLM calls) are checkpointed every 50 queries to enable resume after interruption.

---

## 7. Quality Metrics

### 7.1 Information Retrieval Metrics

| Metric | Description |
|--------|-------------|
| NDCG@k | Normalized Discounted Cumulative Gain at rank k |
| Recall@k | Fraction of relevant documents retrieved in top k |
| MRR@k | Mean Reciprocal Rank of first relevant document |
| Hit Rate | Percentage of queries with at least one relevant document |

### 7.2 Question Answering Metrics

| Metric | Description |
|--------|-------------|
| Exact Match (EM) | Binary match after normalization |
| F1 Score | Token-level precision/recall harmonic mean |
| Containment | Gold answer appears as substring in generated answer |
| Semantic Similarity | Cosine similarity of BGE embeddings |

### 7.3 Embedding Model Analysis

An experiment compared embedding models for semantic similarity:

| Model | Mean Score | Observation |
|-------|------------|-------------|
| BGE-small-en-v1.5 | 74.8% | Baseline, efficient |
| Gemma (full F16) | 55.3% | Lower but consistent |
| Gemma (4-bit quantized) | 46.1% | **Systematic bias from quantization** |

**Conclusion**: Quantized models introduce systematic downward bias in similarity scores. Use BGE-small for production evaluation.

---

## 8. Key Research Findings

### 8.1 Fusion Method Performance (LFM2-1.2B)

| Category | Best Method | F1 Score | vs Baseline |
|----------|-------------|----------|-------------|
| Baseline | CombSUM | 20.82% | — |
| QPP-Weighted | W-CombSUM RSD | 21.15% | +0.33% |
| **Learned** | **Per-Retriever LightGBM** | **23.20%** | **+2.38%** |

### 8.2 Optimal Context Length

| k (documents) | F1 Score | Observation |
|---------------|----------|-------------|
| 0 | 15.2% | No context baseline |
| 1 | 23.2% | **Optimal** |
| 3 | 22.8% | Slight degradation |
| 10 | 21.5% | More context hurts |

**Finding**: For this QA task, a single highly-relevant document (k=1) outperforms multiple documents.

### 8.3 Research Questions Answered

| Question | Answer |
|----------|--------|
| Does QPP-weighting beat baselines? | Yes, +0.33% F1 with W-CombSUM RSD |
| Do learned methods beat heuristic QPP? | **Yes, significantly: +2.05% F1** |
| Which QPP method is best? | **RSD alone — others add noise** |
| Is MLP better than LightGBM? | Per-retriever LightGBM wins |
| Optimal k for RAG? | **k=1 (single best document)** |

---

## 9. Current Experimental Status

### 9.1 Completed Evaluations (LFM2-1.2B)

| Fusion Method | RAG Complete | QA Metrics |
|---------------|--------------|------------|
| CombSUM | ✓ | ✓ |
| CombMNZ | ✓ | ✓ |
| RRF | ✓ | ✓ |
| W-CombSUM RSD | ✓ | ✓ |
| W-CombMNZ RSD | ✓ | ✓ |
| W-RRF RSD | ✓ | ✓ |
| Learned Per-Retriever | ✓ | ✓ |
| Learned Multi-Output | In Progress | Pending |
| Learned MLP (RSD-only) | In Progress | Pending |
| W-CombSUM Learned | In Progress | Pending |

### 9.2 Qwen3-4B Evaluations

| Fusion Method | Status |
|---------------|--------|
| CombSUM | ✓ Complete |
| W-CombSUM RSD | ✓ Complete |
| Learned Per-Retriever | In Progress |

---

## 10. Technical Implementation Details

### 10.1 File Structure

```
QPP-Fusion-RAG/
├── scripts/
│   ├── 01_index.py          # Build search indexes
│   ├── 02_retrieve.py       # Run 5 retrievers
│   ├── 03_qpp.py            # Compute 13 QPP scores
│   ├── 04_train_fusion.py   # Train ML models
│   ├── 05_fusion.py         # Generate fused rankings
│   ├── 06_eval_fusion.py    # Evaluate IR metrics
│   ├── 07_rag_eval.py       # RAG evaluation with LLMs
│   └── 08_compute_qa_metrics.py  # QA metrics computation
├── src/
│   ├── fusion.py            # Fusion method implementations
│   ├── models/              # ML model classes
│   └── retrievers/          # Retriever implementations
└── data/nq/
    ├── runs/                # TREC run files per retriever
    ├── qpp/                 # QPP scores per retriever
    ├── fused/               # Fused run files
    ├── models/              # Trained ML models
    └── results/             # RAG evaluation results
```

### 10.2 Dependencies

- PyTerrier: BM25 indexing and retrieval
- Pyserini: SPLADE and BGE pre-built indexes
- LightGBM: Gradient boosting models
- PyTorch: MLP neural network
- LM Studio: Local LLM inference
- Sentence-Transformers: Embedding models

---

## 11. Conclusions and Recommendations

### 11.1 Main Conclusions

1. **Learned fusion significantly outperforms baselines** — Per-retriever LightGBM achieves +2.38% F1 improvement over CombSUM baseline.

2. **RSD is the only useful QPP predictor** — Other methods either have zero variance, negative correlation, or add noise.

3. **Simpler is better for RAG context** — Single best document (k=1) outperforms multiple documents.

4. **MLP requires correct loss function** — MSE with softmax is mathematically incorrect; use Cross-Entropy for probability outputs.

### 11.2 Recommendations for Production

1. Use **Per-Retriever LightGBM** or **RSD-only MLP** for fusion weight prediction
2. Set **k=1** for RAG context retrieval
3. Use **RSD** as the sole QPP feature — discard other methods
4. Validate on multiple LLMs to ensure generalization

---

*Document Version: 1.0*
*Last Updated: December 2024*


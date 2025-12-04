# QPP-Guided Fusion for Retrieval-Augmented Generation

## Research Goal

**Hypothesis:** Query Performance Prediction (QPP) scores can improve multi-retriever fusion for RAG by dynamically weighting retrievers based on predicted query difficulty.

**Comparison:** Baseline fusion vs QPP-weighted fusion vs Learned fusion across 2 LLMs.

---

## Pipeline Architecture (8 Steps)

### Step 1: Indexing (`01_index.py`) - COMPLETE

Build search indexes for the NQ corpus (2.68M documents).

- PyTerrier BM25 index
- Pyserini SPLADE index (sparse-learned)
- Pyserini BGE FAISS index (dense)

### Step 2: Retrieval (`02_retrieve.py`) - COMPLETE

Run 5 retrievers, generate TREC run files:

| Retriever | Type | Description |

|-----------|------|-------------|

| BM25 | Sparse | Classic lexical matching |

| SPLADE | Sparse-Learned | Neural term expansion |

| BGE | Dense | BAAI embedding similarity |

| BM25→TCT | Sparse→Dense | BM25 + TCT-ColBERT rerank |

| BM25→MonoT5 | Sparse→Neural | BM25 + MonoT5 rerank |

Output: `data/nq/runs/{retriever}.res` (5 files)

### Step 3: QPP Computation (`03_qpp.py`) - COMPLETE

Compute 13 Query Performance Prediction scores per retriever:

| Category | Methods |

|----------|---------|

| Pre-retrieval | AvgIDF, MaxIDF, AvgICTF, MaxVAR, SCQ |

| Post-retrieval | Clarity, NQC, WIG, SMV, RSD, UEF, σmax, n(σ%) |

Output: `data/nq/qpp/{retriever}.qpp` (5 files, 13 scores each)

### Step 4: Train Fusion Models (`04_train_fusion.py`) - COMPLETE

Train ML models to predict optimal retriever weights from QPP features:

| Model | Architecture | Output |

|-------|--------------|--------|

| Per-Retriever LightGBM | 5 separate models | 5 weights |

| Multi-Output LightGBM | Single model | 5 weights |

| MLP | 2-layer neural net | 5 weights |

Output: `data/nq/models/fusion_{type}.pkl` (3 files)

### Step 5: Fusion (`05_fusion.py`) - COMPLETE

Generate 10 fused run files:

| Method | Formula | QPP Used |

|--------|---------|----------|

| CombSUM | Σ scores | None |

| CombMNZ | Σ scores × count | None |

| RRF | Σ 1/(k+rank) | None |

| W-CombSUM RSD | Σ (RSD × score) | RSD only |

| W-CombMNZ RSD | (Σ RSD × score) × count | RSD only |

| W-RRF RSD | Σ RSD/(k+rank) | RSD only |

| Learned Per-Retriever | Σ (w_lgbm × score) | All 13 |

| Learned Multi-Output | Σ (w_lgbm × score) | All 13 |

| Learned MLP | Σ (w_mlp × score) | All 13 |

| W-CombSUM Learned | Σ (w_model × score) | All 13 |

Output: `data/nq/fused/{method}.res` (10 files)

### Step 6: Evaluate Fusion (`06_eval_fusion.py`) - COMPLETE

Compute IR metrics (NDCG@10) for all fusion methods.

Output: `data/nq/fused/comparison_results.json`

### Step 7: RAG Evaluation (`07_rag_eval.py`) - IN PROGRESS

For each fusion × LLM combination:

- Retrieve top-k documents (k=0,1,2,3,4,5,6,10)
- Generate answers using LLM
- Compute retrieval metrics per query

**LLMs:** `liquid/lfm2-1.2b`, `qwen/qwen3-4b-2507`

Output: `data/nq/results/{fusion}__{model}.json` (20 files)

### Step 8: QA Metrics (`08_compute_qa_metrics.py`) - IN PROGRESS

Compute answer quality metrics against gold answers:

| Metric | Description |

|--------|-------------|

| EM% | Exact match after normalization |

| F1% | Token-level F1 score |

| Containment% | Gold answer is substring of output |

| Semantic% | Cosine similarity via BGE embeddings |

Output: Metrics added to result JSON files

---

## Current Execution Status

### RAG Evaluations (20 total)

| # | Fusion | LFM2-1.2B | Qwen3-4B |
|---|--------|-----------|----------|
| 1 | combsum | ✓ DONE | ⏳ 30% |
| 2 | combmnz | ⏳ RUNNING | ○ PENDING |
| 3 | rrf | ○ PENDING | ○ PENDING |
| 4 | wcombsum_rsd | ✓ DONE | ✓ DONE |
| 5 | wcombmnz_rsd | ○ PENDING | ○ PENDING |
| 6 | wrrf_rsd | ○ PENDING | ○ PENDING |
| 7 | learned_per_retriever | ○ PENDING | ○ PENDING |
| 8 | learned_multioutput | ○ PENDING | ○ PENDING |
| 9 | learned_mlp | ○ PENDING | ○ PENDING |
| 10 | wcombsum_learned | ○ PENDING | ○ PENDING |

**Complete:** 4/20 | **In Progress:** 2/20 | **Pending:** 14/20

### QA Metrics Status

- combsum + LFM: ✓ All 4 metrics (EM/F1/Containment/Semantic)
- wcombsum_rsd + LFM: ✓ All 4 metrics
- wcombsum_rsd + Qwen: ✓ All 4 metrics
- Others: Pending (run after RAG completes)

### Early Results: Baseline vs QPP-Weighted (LFM)

| Metric | CombSUM (baseline) | W-CombSUM RSD (QPP) | Δ |
|--------|-------------------|---------------------|---|
| EM% (k=1) | 0.70 | 0.81 | +0.11 |
| F1% (k=1) | 20.82 | 20.98 | +0.16 |
| Containment% (k=10) | 48.68 | 49.01 | +0.33 |
| Semantic% (k=10) | 67.17 | 67.22 | +0.05 |

**Finding:** QPP-weighted fusion shows small but consistent improvement over baseline.

---

## Output Files

```
data/nq/results/
├── _SCHEMA.json                              # JSON structure docs
├── combsum__liquid_lfm2-1.2b.json           # Result files (20)
├── combsum__qwen_qwen3-4b-2507.json
├── combmnz__liquid_lfm2-1.2b.json
├── ... (16 more)
└── checkpoint_{fusion}__{model}.json         # Resume checkpoints
```

---

## Final Deliverables

1. **Comparison Table:** All 20 runs × 4 QA metrics × 8 k-values
2. **Research Findings:**

   - Does QPP-weighting beat baselines?
   - Do learned methods beat heuristic QPP?
   - Which LLM performs better?
   - Optimal k value for RAG context

3. **Best Configuration:** Fusion method + LLM + k recommendation
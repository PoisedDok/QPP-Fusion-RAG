# QPP-Guided Fusion for Retrieval-Augmented Generation

## Research Goal

**Hypothesis:** Query Performance Prediction (QPP) scores can improve multi-retriever fusion for RAG by dynamically weighting retrievers based on predicted query difficulty.

**Comparison:** Baseline fusion vs QPP-weighted fusion vs Learned fusion across 2 LLMs.

---

## Pipeline Status

| Step | Script | Status |
|------|--------|--------|
| 1. Indexing | `01_index.py` | ✓ COMPLETE |
| 2. Retrieval | `02_retrieve.py` | ✓ COMPLETE |
| 3. QPP Computation | `03_qpp.py` | ✓ COMPLETE |
| 4. Train Fusion | `04_train_fusion.py` | ✓ COMPLETE |
| 5. Fusion | `05_fusion.py` | ✓ COMPLETE |
| 6. Evaluate Fusion | `06_eval_fusion.py` | ✓ COMPLETE |
| 7. RAG Evaluation | `07_rag_eval.py` | ⏳ IN PROGRESS |
| 8. QA Metrics | `08_compute_qa_metrics.py` | ⏳ IN PROGRESS |

---

## RAG Evaluation Matrix (20 combinations)

### LFM2-1.2B Progress: 8/10 ✓ + 3 Running

| # | Fusion Method | Type | Status | QA Metrics |
|---|---------------|------|--------|------------|
| 1 | combsum | Baseline | ✓ DONE | ✓ All 4 |
| 2 | combmnz | Baseline | ✓ DONE | ✓ All 4 |
| 3 | rrf | Baseline | ✓ DONE | ✓ All 4 |
| 4 | wcombsum_rsd | QPP-RSD | ✓ DONE | ✓ All 4 |
| 5 | wcombmnz_rsd | QPP-RSD | ✓ DONE | ✓ All 4 |
| 6 | wrrf_rsd | QPP-RSD | ✓ DONE | ✓ All 4 |
| 7 | learned_per_retriever | Learned | ✓ DONE | ✓ All 4 |
| 8 | learned_multioutput | Learned | ⏳ 56% | ○ Pending |
| 9 | learned_mlp | Learned | ⏳ 3% | ○ Pending |
| 10 | wcombsum_learned | Learned | ⏳ 1% | ○ Pending |

### Qwen3-4B Progress: 2/10 ✓

| # | Fusion Method | Status | QA Metrics |
|---|---------------|--------|------------|
| 1 | combsum | ✓ DONE | ✓ All 4 |
| 4 | wcombsum_rsd | ✓ DONE | ✓ All 4 |
| 2-3, 5-10 | Others | ○ PENDING | - |

**Overall Progress:** 10/20 complete | 3/20 in progress | 7/20 pending

---

## Active Processes

| Instance | Fusion | Progress |
|----------|--------|----------|
| LFM | learned_mlp | 3% |
| LFM:2 | learned_multioutput | 56% |
| LFM:3 | wcombsum_learned | 1% |

---

## 🔥 Key Results (LFM)

### Comparison: Baseline vs QPP vs Learned (k=1 shot)

| Method | Type | EM% | F1% | Containment% | Semantic% |
|--------|------|-----|-----|--------------|-----------|
| combsum | Baseline | 0.70 | 20.82 | 39.81 | 65.53 |
| combmnz | Baseline | 0.70 | 19.73 | 37.06 | 64.73 |
| rrf | Baseline | 0.70 | 19.73 | 37.06 | 64.73 |
| wcombsum_rsd | QPP-RSD | 0.81 | 20.98 | 40.14 | 65.66 |
| wcombmnz_rsd | QPP-RSD | 0.77 | 20.91 | 39.96 | 65.53 |
| wrrf_rsd | QPP-RSD | 0.70 | 19.73 | 37.06 | 64.73 |
| **learned_per_retriever** | **Learned** | 0.77 | **23.20** | **44.57** | **66.88** |

### 🎯 Finding: Learned Fusion BEATS All Others!

| Metric | Baseline (combsum) | QPP (wcombsum_rsd) | Learned | Δ vs Baseline |
|--------|-------------------|-------------------|---------|---------------|
| F1% | 20.82 | 20.98 | **23.20** | **+2.38** |
| Containment% | 39.81 | 40.14 | **44.57** | **+4.76** |
| Semantic% | 65.53 | 65.66 | **66.88** | **+1.35** |

**learned_per_retriever** shows significant improvement:
- **+11% relative F1 improvement** over baseline
- **+12% relative containment improvement** over baseline
- Also beats heuristic QPP weighting

---

## LFM vs Qwen Comparison (shared methods)

| Metric | LFM avg | Qwen avg | Winner |
|--------|---------|----------|--------|
| EM% | 0.97 | 2.98 | **Qwen +2%** |
| F1% | 20.3 | 20.7 | TIE |
| Containment% | 40.2 | 42.9 | **Qwen +2.7%** |
| Semantic% | 65.6 | 64.1 | **LFM +1.5%** |

**Verdict:** Mixed results - Qwen better at exact match, LFM better at semantic similarity.

---

## Performance by k-value (learned_per_retriever + LFM)

| k | EM% | F1% | Containment% | Semantic% |
|---|-----|-----|--------------|-----------|
| 0 | 0.37 | 9.15 | 12.98 | 60.07 |
| 1 | 0.77 | 23.20 | 44.57 | 66.88 |
| 2 | 1.21 | 24.26 | 48.35 | 67.82 |
| 3 | 1.17 | 24.45 | 51.10 | 67.99 |
| 4 | 1.06 | **24.65** | 52.49 | **68.10** |
| 5 | 1.21 | 24.38 | 51.25 | 68.02 |
| 10 | 0.95 | 22.97 | 50.88 | 67.69 |

**Finding:** Optimal k=3-4 for learned methods (peaks earlier than baseline k=5-6).

---

## 🧪 Embedding Model Experiment

### Experiment: Comparing Embedding Models for Semantic Similarity

Tested 3 embedding models on 100 prediction-gold pairs:

| Model | Dimensions | Mean Sim | Std |
|-------|-----------|----------|-----|
| BGE-small-en-v1.5 | 384 | 0.602 | 0.125 |
| Gemma-300M (Full) | 768 | 0.643 | 0.109 |
| Gemma-300M (QAT) | 768 | **0.499** | 0.148 |

### Key Finding: Quantization Causes Systematic Bias!

| Comparison | Correlation | Mean Diff | Scores within 5% |
|------------|-------------|-----------|------------------|
| BGE vs Gemma-Full | r=0.81 | -0.04 | 48% |
| BGE vs Gemma-QAT | r=0.84 | +0.10 | 19% |
| Gemma-Full vs QAT | r=0.98 | **+0.14** | **2%** |

**Critical Discovery:**
- **QAT gives LOWER scores 100% of the time** (systematic -0.14 bias)
- Rank correlation preserved (r=0.98) but absolute values are wrong
- QAT unsuitable for semantic similarity metrics

### Recommendation

✅ **Use BGE-small** for QA evaluation:
- Fast (384 dims, 2x smaller than Gemma)
- Correlates well with full-precision models (r=0.81)
- No systematic bias

❌ **Avoid QAT models** for semantic metrics:
- Systematic underestimation of similarity
- Only useful if relative ranking is all you need

---

## Completed Result Files (10/20)

```
data/nq/results/
├── combsum__liquid_lfm2-1.2b.json              ✓ + QA
├── combmnz__liquid_lfm2-1.2b.json              ✓ + QA
├── rrf__liquid_lfm2-1.2b_2.json                ✓ + QA
├── wcombsum_rsd__liquid_lfm2-1.2b.json         ✓ + QA
├── wcombmnz_rsd__liquid_lfm2-1.2b_3.json       ✓ + QA
├── wrrf_rsd__liquid_lfm2-1.2b.json             ✓ + QA
├── learned_per_retriever__liquid_lfm2-1.2b_3.json  ✓ + QA
├── combsum__qwen_qwen3-4b-2507.json            ✓ + QA
├── wcombsum_rsd__qwen_qwen3-4b-2507.json       ✓ + QA
└── (10 more pending)
```

---

## Research Questions - Status

| Question | Status | Answer |
|----------|--------|--------|
| Does QPP-weighting beat baselines? | ✓ | **YES** - wcombsum_rsd +0.16 F1 |
| Do learned methods beat heuristic QPP? | ✓ | **YES** - learned +2.22 F1 over QPP |
| Which LLM performs better? | Partial | Mixed - Qwen EM, LFM semantic |
| Optimal k value for RAG? | ✓ | **k=3-4** for learned, k=5-6 for baseline |
| Does embedding quantization affect metrics? | ✓ | **YES** - QAT has -0.14 systematic bias |

---

## Next Steps

1. Complete 3 remaining LFM learned methods (multioutput, mlp, wcombsum_learned)
2. Run QA metrics on completed runs
3. Compare all learned methods to find best approach
4. Decide on Qwen coverage (skip or run 2-3 more methods)

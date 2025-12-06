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
| 7. RAG Evaluation | `07_rag_eval.py` | ✓ COMPLETE (LFM) |
| 8. QA Metrics | `08_compute_qa_metrics.py` | ✓ COMPLETE (LFM) |
| 9. Analysis | `09_analyze_results.py` | ✓ COMPLETE |

---

## Evaluation Progress

### LFM2-1.2B: ✓ 10/10 COMPLETE

| # | Fusion Method | Type | Status |
|---|---------------|------|--------|
| 1-3 | combsum, combmnz, rrf | Baseline | ✓ |
| 4-6 | wcombsum_rsd, wcombmnz_rsd, wrrf_rsd | QPP-RSD | ✓ |
| 7-10 | learned_per_retriever, learned_multioutput, learned_mlp, wcombsum_learned | Learned | ✓ |

### Qwen3-4B: 2/10

| # | Fusion Method | Status |
|---|---------------|--------|
| 1 | combsum | ✓ |
| 4 | wcombsum_rsd | ✓ |
| Others | - | ○ PENDING |

**Total: 12/20 complete**

---

## Key Findings

### Hypothesis Validation

| Question | Answer |
|----------|--------|
| Does QPP-weighting beat baselines? | **MARGINAL** (+0.2% Recall) |
| Do learned methods beat QPP-weighted? | **YES** (+14.7% Recall, +4% F1) |
| Best method? | **learned_multioutput** |
| Optimal k? | k=3-4 for QA metrics |

### Rankings @ k=5

1. **learned_multioutput** - 67.59% Recall, 24.28% F1
2. learned_per_retriever - 67.28% Recall
3. wcombsum_learned - 67.28% Recall, 24.49% F1
4. learned_mlp - 64.93% Recall
5. wcombsum_rsd (best QPP) - 58.95% Recall
6. combsum (best baseline) - 58.84% Recall

**Conclusion:** Learned methods dominate. Simple QPP weighting provides negligible improvement.

---

## Result Files

```
data/nq/results/lfm-NQ/           # 10 result JSONs
data/nq/results/lfm-NQ/analysis/  # Report + visualizations
data/nq/results/qwen-NQ/          # 2 result JSONs
```

---

## Next Steps

1. ~~Complete LFM experiments~~ ✓
2. ~~Run analysis~~ ✓
3. Decide on Qwen coverage (optional)
4. Write up for thesis

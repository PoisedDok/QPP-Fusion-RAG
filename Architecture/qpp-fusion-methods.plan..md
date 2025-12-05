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

### LFM2-1.2B Progress: 7/10 ✓ + 3 Running

| # | Fusion Method | Status | QA Metrics |
|---|---------------|--------|------------|
| 1 | combsum | ✓ DONE | ✓ All 4 |
| 2 | combmnz | ✓ DONE | ✓ All 4 |
| 3 | rrf | ✓ DONE | ✓ All 4 |
| 4 | wcombsum_rsd | ✓ DONE | ✓ All 4 |
| 5 | wcombmnz_rsd | ✓ DONE | ✓ All 4 |
| 6 | wrrf_rsd | ✓ DONE | ✓ All 4 |
| 7 | learned_per_retriever | ✓ DONE | ⏳ Running |
| 8 | learned_multioutput | ⏳ 55% | ○ Pending |
| 9 | learned_mlp | ⏳ 2% | ○ Pending |
| 10 | wcombsum_learned | ⏳ 0% | ○ Pending |

### Qwen3-4B Progress: 2/10 ✓

| # | Fusion Method | Status | QA Metrics |
|---|---------------|--------|------------|
| 1 | combsum | ✓ DONE | ✓ All 4 |
| 2 | combmnz | ○ PENDING | - |
| 3 | rrf | ○ PENDING | - |
| 4 | wcombsum_rsd | ✓ DONE | ✓ All 4 |
| 5 | wcombmnz_rsd | ○ PENDING | - |
| 6 | wrrf_rsd | ○ PENDING | - |
| 7 | learned_per_retriever | ○ PENDING | - |
| 8 | learned_multioutput | ○ PENDING | - |
| 9 | learned_mlp | ○ PENDING | - |
| 10 | wcombsum_learned | ○ PENDING | - |

**Overall Progress:** 9/20 complete | 4/20 in progress | 7/20 pending

---

## Active Processes

| Instance | Fusion | Progress |
|----------|--------|----------|
| LFM | learned_mlp | 2% |
| LFM:2 | learned_multioutput | 55% |
| LFM:3 | wcombsum_learned | 0% |
| Embed:2 | QA metrics for learned_per_retriever | Running |

---

## Result File Schema

All result files follow this standardized schema:

```json
{
  "_metadata": {
    "created": "2025-12-05T...",
    "script": "07_rag_eval.py",
    "fusion_method": "combsum",
    "model": "liquid/lfm2-1.2b"
  },
  "config": {
    "model": "...",
    "k_values": [0, 1, 2, 3, 4, 5, 6, 10],
    "run_path": "..."
  },
  "summary": {
    "retrieval_metrics_by_k": {
      "0": {"recall_at_k": 0.0, "mrr@k": 0.0, "hit_rate": 0.0, "n": 3452},
      "1": {"recall_at_k": 0.31, "mrr@k": 0.31, "hit_rate": 0.31, "n": 3452}
    },
    "qa_metrics_by_k": {
      "0": {"em": 0.29, "f1": 8.85, "containment": 12.57, "semantic": 59.98, "n_matched": 2728},
      "1": {"em": 0.70, "f1": 20.82, "containment": 39.81, "semantic": 65.53, "n_matched": 2728}
    }
  },
  "results": [
    {
      "qid": "...",
      "query": "...",
      "shots": {
        "0": {"k": 0, "answer": "...", "em": 0, "f1": 0.1, "containment": 0, "semantic": 0.6, "reciprocal_rank": 0.0, ...},
        "1": {"k": 1, "answer": "...", "em": 0, "f1": 0.2, "containment": 1, "semantic": 0.7, "reciprocal_rank": 1.0, ...}
      }
    }
  ]
}
```

---

## QA Metrics

| Metric | Description | Script |
|--------|-------------|--------|
| EM% | Exact match after normalization | 08_compute_qa_metrics.py |
| F1% | Token-level F1 score | 08_compute_qa_metrics.py |
| Containment% | Gold answer substring of output | 08_compute_qa_metrics.py |
| Semantic% | Cosine similarity (BGE embeddings) | 08_compute_qa_metrics.py |
| MRR@k | Mean Reciprocal Rank | 07_rag_eval.py |

---

## Results Summary (LFM completed)

### Baselines vs QPP-Weighted (k=1 shot)

| Method | Type | EM% | F1% | Containment% | Semantic% |
|--------|------|-----|-----|--------------|-----------|
| combsum | Baseline | 0.70 | 20.82 | 39.81 | 65.53 |
| combmnz | Baseline | 0.70 | 19.73 | 37.06 | 64.73 |
| rrf | Baseline | 0.70 | 19.73 | 37.06 | 64.73 |
| wcombsum_rsd | QPP-RSD | 0.81 | 20.98 | 40.14 | 65.66 |
| wcombmnz_rsd | QPP-RSD | 0.77 | 20.91 | 39.96 | 65.53 |
| wrrf_rsd | QPP-RSD | 0.70 | 19.73 | 37.06 | 64.73 |

**Finding:** QPP-weighted CombSUM shows +0.11 EM improvement over baseline.

### Performance by k-value (wcombsum_rsd + LFM)

| k | EM% | F1% | Containment% | Semantic% |
|---|-----|-----|--------------|-----------|
| 0 | 0.29 | 8.81 | 12.50 | 59.95 |
| 1 | 0.81 | 20.98 | 40.14 | 65.66 |
| 2 | 0.99 | 22.35 | 44.61 | 66.58 |
| 3 | 1.06 | 22.81 | 46.08 | 66.94 |
| 5 | 1.21 | 22.77 | 47.62 | 67.06 |
| 10 | 1.21 | 22.50 | 49.01 | 67.22 |

**Finding:** Performance improves with more context, plateaus around k=5-6.

---

## Completed Result Files

```
data/nq/results/
├── combsum__liquid_lfm2-1.2b.json          ✓
├── combmnz__liquid_lfm2-1.2b.json          ✓
├── rrf__liquid_lfm2-1.2b_2.json            ✓
├── wcombsum_rsd__liquid_lfm2-1.2b.json     ✓
├── wcombmnz_rsd__liquid_lfm2-1.2b_3.json   ✓
├── wrrf_rsd__liquid_lfm2-1.2b.json         ✓
├── learned_per_retriever__liquid_lfm2-1.2b_3.json  ✓ (QA running)
├── combsum__qwen_qwen3-4b-2507.json        ✓
├── wcombsum_rsd__qwen_qwen3-4b-2507.json   ✓
└── (11 more pending)
```

---

## Research Questions

1. **Does QPP-weighting beat baselines?** → Early signs: YES (+0.11 EM)
2. **Do learned methods beat heuristic QPP?** → Pending (3 learned methods running)
3. **Which LLM performs better?** → Pending (need more Qwen results)
4. **Optimal k value for RAG?** → k=5-6 appears optimal

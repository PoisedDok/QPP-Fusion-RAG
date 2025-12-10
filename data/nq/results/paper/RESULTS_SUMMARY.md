# NQ Retrieval Results - BEIR Comparison

**Generated:** 2025-12-09 16:26  
**Queries:** 3452 test queries

---

## Summary

| Metric | Best Method | Score | BEIR Best | Δ |
|--------|-------------|-------|-----------|---|
| nDCG@10 | Splade | **0.5373** | BM25+CE (0.533) | +0.8% |

---

## Individual Ranker Performance

| Method | Type | nDCG@5 | nDCG@10 | nDCG@20 | MRR@10 | R@100 |
|--------|------|--------|---------|---------|--------|-------|
| Splade | Sparse | 0.4999 | **0.5373** | 0.5609 | 0.4876 | 0.9296 |
| BM25_MonoT5 | Rerank | 0.4900 | **0.5153** | 0.5271 | 0.4820 | 0.7387 |
| BGE | Dense | 0.4679 | **0.5106** | 0.5340 | 0.4596 | 0.9268 |
| BM25 | Lexical | 0.2653 | **0.3044** | 0.3316 | 0.2624 | 0.7497 |

---

## BEIR Comparison

| Our Method | Type | Ours | BEIR Method | BEIR Score | Δ% |
|------------|------|------|-------------|------------|-----|
| Splade | Sparse | 0.5373 | docT5query | 0.399 | +34.7% |
| BM25_MonoT5 | Rerank | 0.5153 | BM25+CE | 0.533 | -3.3% |
| BGE | Dense | 0.5106 | TAS-B | 0.463 | +10.3% |
| BM25 | Lexical | 0.3044 | BM25 | 0.329 | -7.5% |

†: In-domain trained model

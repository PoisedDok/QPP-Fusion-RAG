# HOTPOTQA Retrieval Results - BEIR Comparison

**Generated:** 2025-12-10 05:41  
**Queries:** 7405 test queries

---

## Summary

| Metric | Best Method | Score | BEIR Best | Δ |
|--------|-------------|-------|-----------|---|
| nDCG@10 | BM25_MonoT5 | **0.7136** | BM25+CE (0.707) | +0.9% |

---

## Individual Ranker Performance

| Method | Type | nDCG@5 | nDCG@10 | nDCG@20 | MRR@10 | R@100 |
|--------|------|--------|---------|---------|--------|-------|
| BM25_MonoT5 | Rerank | 0.6980 | **0.7136** | 0.7206 | 0.8797 | 0.7701 |
| Splade | Sparse | 0.6676 | **0.6868** | 0.6997 | 0.8655 | 0.8177 |
| BGE | Dense | 0.6399 | **0.6579** | 0.6694 | 0.7874 | 0.7943 |
| BM25_TCT | Late-Int | 0.5891 | **0.6117** | 0.6269 | 0.7862 | 0.7701 |
| BM25 | Lexical | 0.5618 | **0.5858** | 0.6023 | 0.7505 | 0.7701 |

---

## BEIR Comparison

| Our Method | Type | Ours | BEIR Method | BEIR Score | Δ% |
|------------|------|------|-------------|------------|-----|
| BM25_MonoT5 | Rerank | 0.7136 | BM25+CE | 0.707 | +0.9% |
| Splade | Sparse | 0.6868 | docT5query | 0.580 | +18.4% |
| BGE | Dense | 0.6579 | TAS-B | 0.584 | +12.7% |
| BM25_TCT | Late-Int | 0.6117 | ColBERT | 0.593 | +3.2% |
| BM25 | Lexical | 0.5858 | BM25 | 0.603 | -2.9% |

†: In-domain trained model

# Natural Questions (NQ) Dataset Setup

## Overview

Natural Questions: Real Google queries with Wikipedia answers (Kwiatkowski et al., 2019)
- **Test queries:** 3,452 (with gold answers)
- **Corpus:** 2.7M Wikipedia paragraphs
- **Question type:** Factoid, single-hop

## Directory Structure

```
nq/
├── BEIR-nq/
│   ├── corpus.jsonl           (1.0 GB - 2.7M documents) [LFS]
│   ├── queries.jsonl          (800 KB - 3.4K queries)
│   ├── qrels/test.tsv         (relevance judgments)
│   └── nq_gold_answers.json   (extracted from HuggingFace)
├── runs/                       (retrieval results) [LFS]
│   ├── BM25.res / BM25.norm.res
│   ├── BGE.res / BGE.norm.res
│   ├── Splade.res / Splade.norm.res
│   ├── BM25_TCT.res / BM25_TCT.norm.res
│   └── BM25_MonoT5.res / BM25_MonoT5.norm.res
├── qpp/                        (QPP scores per retriever)
│   ├── BM25.res.mmnorm.qpp
│   ├── BGE.res.mmnorm.qpp
│   ├── Splade.res.mmnorm.qpp
│   ├── BM25_TCT.res.mmnorm.qpp
│   └── BM25_MonoT5.res.mmnorm.qpp
├── models/                     (trained fusion models)
│   ├── fusion_mlp.pkl
│   ├── fusion_per_retriever.pkl
│   ├── fusion_multioutput.pkl
│   └── fusion_weights_model.pkl
├── fused/                      (fused rankings) [LFS]
│   ├── combsum.res, combmnz.res, rrf.res
│   ├── wcombsum_rsd.res, wcombmnz_rsd.res, wrrf_rsd.res
│   ├── wcombsum_learned.res
│   ├── learned_mlp.res
│   ├── learned_per_retriever.res
│   └── learned_multioutput.res
├── results/                    (RAG evaluation outputs)
│   ├── _SCHEMA.json
│   ├── lfm-NQ/*.json          (Liquid LFM results)
│   ├── qwen-NQ/*.json         (Qwen results)
│   └── checkpoint_*.json       (ongoing evaluations)
├── cache/                      (HuggingFace downloads, embeddings)
└── index/pyterrier/            (BM25 index)
```

## Gold Answers

**Source:** HuggingFace `natural_questions` dataset (validation split)
**Format:** JSON dictionary
```json
{
  "what is non controlling interest on balance sheet": ["minority interest", "non-controlling interest"],
  "who won the 2020 election": ["Joe Biden"]
}
```

**Statistics:**
- Total: ~3,400 unique questions
- Multi-answer: ~40% (multiple valid answer strings)
- Source: Real Google search queries

**Extraction:**
```python
# Automatic on first run via scripts/08_compute_qa_metrics.py
# Downloads ~40GB HuggingFace dataset, extracts answers, caches locally
```

## Pipeline Commands

```bash
# Already complete - all files present
# Re-run specific steps as needed:

# Re-compute QPP
python scripts/03_qpp.py --runs_dir data/nq/runs

# Re-train fusion models
python scripts/04_train_fusion.py --corpus_path /path/to/beir/nq

# Re-run RAG evaluation
python scripts/07_rag_eval.py \
    --corpus_path /path/to/beir/nq \
    --model "liquid/lfm2-1.2b"

# Compute QA metrics
python scripts/08_compute_qa_metrics.py \
    --results_file data/nq/results/*.json \
    --dataset nq \
    --metrics em,f1,containment,semantic
```

## Comparison to HotpotQA

| Metric | NQ | HotpotQA |
|--------|-----|----------|
| **Test Queries** | 3,452 | 7,405 |
| **Corpus Size** | 2.7M docs | 5.2M docs |
| **Question Type** | Factoid, single-hop | Multi-hop reasoning |
| **Answer Length** | 1-5 words | Variable (yes/no + spans) |
| **Retrieval Difficulty** | Medium | Hard (need 2+ docs) |
| **Gold Answer Source** | HuggingFace (40GB) | BEIR metadata |

## Notes

- **Git LFS:** Large files tracked (`corpus.jsonl`, `.res` files)
- **All generated files tracked:** Models, results, QPP scores committed for reproducibility
- **Gold answers cached:** First run downloads 40GB, subsequent runs use cache
- **Status:** Operational - full pipeline executed, results available

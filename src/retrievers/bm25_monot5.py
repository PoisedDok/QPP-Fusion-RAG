"""
Incoming: query, index, corpus_path --- {str, pt.Index, str}
Processing: two-stage retrieval --- {PyTerrier pipeline: BM25 >> CrossEncoder}
Outgoing: ranked results --- {RetrieverResult}

BM25 >> MonoT5/CrossEncoder Hybrid Retriever (PyTerrier Pipeline)
-----------------------------------------------------------------
Uses PyTerrier pipeline composition for two-stage retrieval:
1. BM25 first-stage recall
2. CrossEncoder reranking (ms-marco-MiniLM-L-6-v2)

STRICT: Uses sentence_transformers CrossEncoder. No manual fallbacks.
"""

import os
import time
from typing import Dict

import pandas as pd
import numpy as np
from sentence_transformers import CrossEncoder

from .base import BaseRetriever, RetrieverResult


def _ensure_pyterrier_init():
    """Lazy PyTerrier initialization to avoid JVM conflicts with pyserini."""
    import pyterrier as pt
    if hasattr(pt, 'java') and hasattr(pt.java, 'init') and not pt.started():
        pt.java.init()
    elif not pt.started():
        pt.init()
    return pt


class CrossEncoderReranker:
    """
    PyTerrier Transformer wrapper for CrossEncoder reranking.
    
    Takes a DataFrame with 'query' and 'text' columns,
    scores each (query, text) pair, and returns reranked results.
    """
    
    def __init__(self, model_name: str, batch_size: int = 256, device: str = None):
        self.model_name = model_name
        self.batch_size = batch_size
        self.device = device or ("mps" if os.uname().sysname == "Darwin" else "cuda")
        
        print(f"[CrossEncoderReranker] Loading {model_name}...")
        self.model = CrossEncoder(model_name, device=self.device)
    
    def transform(self, topics_and_docs: pd.DataFrame) -> pd.DataFrame:
        """Rerank documents using CrossEncoder."""
        if len(topics_and_docs) == 0:
            return topics_and_docs
        
        # Build (query, doc_text) pairs
        pairs = [
            [row["query"], row.get("text", "")]
            for _, row in topics_and_docs.iterrows()
        ]
        
        # Score pairs
        scores = self.model.predict(pairs, batch_size=self.batch_size, show_progress_bar=False)
        
        # Update scores
        result = topics_and_docs.copy()
        result["score"] = scores
        
        # Rerank within each query
        result = result.sort_values(["qid", "score"], ascending=[True, False])
        
        # Reset ranks
        result["rank"] = result.groupby("qid").cumcount() + 1
        
        return result


class BM25MonoT5Retriever(BaseRetriever):
    """
    Two-stage retrieval using PyTerrier pipeline composition.
    
    Pipeline: BM25 >> text_loader >> CrossEncoder
    
    Despite the name "MonoT5", uses the faster ms-marco-MiniLM CrossEncoder
    which provides similar quality with 10x speed improvement.
    """
    
    name = "BM25_MonoT5"
    CE_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    
    def __init__(
        self,
        index_path: str,
        corpus_path: str,
        first_stage_k: int = 100,
        batch_size: int = 256
    ):
        """
        Initialize BM25>>CrossEncoder pipeline.
        
        Args:
            index_path: Path to PyTerrier index
            corpus_path: Path to BEIR corpus directory
            first_stage_k: Number of candidates from BM25
            batch_size: Batch size for CrossEncoder
        """
        pt = _ensure_pyterrier_init()
        
        self.corpus_path = corpus_path
        self.first_stage_k = first_stage_k
        self.batch_size = batch_size
        
        # BM25 first stage
        self.index = pt.IndexFactory.of(index_path)
        self.bm25 = pt.BatchRetrieve(
            self.index,
            wmodel="BM25",
            num_results=first_stage_k,
            metadata=["docno"]
        )
        
        # CrossEncoder reranker
        self.reranker = CrossEncoderReranker(
            model_name=self.CE_MODEL,
            batch_size=batch_size
        )
        
        # Build offset index for lazy corpus loading
        self._corpus_offsets = self._build_corpus_offsets()
        self._corpus_cache = {}  # Cache for loaded docs
        
        # Build a manual pipeline (not >> operator) to avoid pt.Transformer dependency
        print(f"[BM25_MonoT5] Pipeline ready: BM25(k={first_stage_k}) >> CrossEncoder (lazy loading)")
    
    def _build_corpus_offsets(self) -> Dict[str, int]:
        """Build doc_id -> byte offset map for lazy loading."""
        import json
        
        corpus_file = os.path.join(self.corpus_path, "corpus.jsonl")
        offsets = {}
        
        print(f"[BM25_MonoT5] Building corpus offset index (lazy loading)...")
        with open(corpus_file, 'r', encoding='utf-8') as f:
            offset = 0
            for line in f:
                doc = json.loads(line)
                doc_id = doc.get("_id", "")
                offsets[doc_id] = offset
                offset += len(line.encode('utf-8'))
        
        print(f"[BM25_MonoT5] Indexed {len(offsets)} documents (0 loaded in RAM)")
        return offsets
    
    def _load_docs(self, doc_ids: list) -> Dict[str, str]:
        """Load specific docs by ID on-demand."""
        import json
        
        corpus_file = os.path.join(self.corpus_path, "corpus.jsonl")
        texts = {}
        
        # Sort by offset for sequential reads
        ids_with_offset = [(did, self._corpus_offsets.get(did, -1)) for did in doc_ids]
        ids_with_offset = [(did, off) for did, off in ids_with_offset if off >= 0]
        ids_with_offset.sort(key=lambda x: x[1])
        
        with open(corpus_file, 'r', encoding='utf-8') as f:
            for doc_id, offset in ids_with_offset:
                f.seek(offset)
                doc = json.loads(f.readline())
                text = (doc.get("title", "") + " " + doc.get("text", "")).strip()
                texts[doc_id] = text
        
        return texts
    
    def _add_text(self, df):
        """PyTerrier transformer to add text column (lazy-loaded on demand)."""
        import pandas as pd
        
        df = df.copy()
        
        # Get unique doc IDs needed for this batch
        doc_ids = df["docno"].unique().tolist()
        doc_ids_str = [str(d) for d in doc_ids]
        
        # Load only the docs we need (not in cache)
        needed_ids = [d for d in doc_ids_str if d not in self._corpus_cache]
        if needed_ids:
            loaded = self._load_docs(needed_ids)
            self._corpus_cache.update(loaded)
        
        # Add text column
        df["text"] = df["docno"].apply(lambda x: self._corpus_cache.get(str(x), ""))
        return df
    
    def _run_pipeline(self, query_df: pd.DataFrame) -> pd.DataFrame:
        """Run the manual BM25 >> text >> reranker pipeline."""
        # Step 1: BM25 retrieval
        results_df = self.bm25.transform(query_df)
        
        # Step 2: Add text
        results_df = self._add_text(results_df)
        
        # Step 3: CrossEncoder reranking
        results_df = self.reranker.transform(results_df)
        
        return results_df
    
    def retrieve(
        self,
        query: str,
        qid: str,
        top_k: int = 100,
        **kwargs
    ) -> RetrieverResult:
        """Retrieve using manual pipeline: BM25 >> text >> CrossEncoder."""
        import re
        
        start = time.time()
        
        # Sanitize query
        clean_query = re.sub(r"[^a-zA-Z0-9\s]", " ", query)
        clean_query = re.sub(r"\s+", " ", clean_query).strip()
        
        query_df = pd.DataFrame([{"qid": qid, "query": clean_query}])
        
        # Run pipeline
        results_df = self._run_pipeline(query_df)
        results_df = results_df.head(top_k)
        
        results = []
        for rank, (_, row) in enumerate(results_df.iterrows(), start=1):
            results.append((
                str(row["docno"]),
                float(row["score"]),
                rank
            ))
        
        latency = (time.time() - start) * 1000
        
        return RetrieverResult(
            qid=qid,
            results=results,
            retriever_name=self.name,
            latency_ms=latency,
            metadata={"model": self.CE_MODEL, "first_stage_k": self.first_stage_k}
        )
    
    def retrieve_batch(
        self,
        queries: Dict[str, str],
        top_k: int = 100,
        **kwargs
    ) -> Dict[str, RetrieverResult]:
        """Batch retrieval using manual pipeline: BM25 >> text >> CrossEncoder."""
        import re
        
        start = time.time()
        n_queries = len(queries)
        
        print(f"[BM25_MonoT5] Batch retrieval: {n_queries} queries")
        
        # Build query DataFrame
        query_df = pd.DataFrame([
            {
                "qid": qid,
                "query": re.sub(r"\s+", " ", re.sub(r"[^a-zA-Z0-9\s]", " ", text)).strip()
            }
            for qid, text in queries.items()
        ])
        
        # Run pipeline
        results_df = self._run_pipeline(query_df)
        
        # Group by query and build results (OPTIMIZED: O(n) instead of O(n²))
        results = {}
        grouped = results_df.groupby("qid")
        for qid, group in grouped:
            qid_results = group.head(top_k)
            
            doc_list = []
            for rank, (_, row) in enumerate(qid_results.iterrows(), start=1):
                doc_list.append((
                    str(row["docno"]),
                    float(row["score"]),
                    rank
                ))
            
            results[qid] = RetrieverResult(
                qid=qid,
                results=doc_list,
                retriever_name=self.name,
                latency_ms=0,
                metadata={"model": self.CE_MODEL}
            )
        
        total_time = (time.time() - start) * 1000
        print(f"[BM25_MonoT5] Batch complete: {n_queries} queries in {total_time:.1f}ms")
        
        return results

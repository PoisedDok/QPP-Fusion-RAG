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

# STRICT: Fail immediately if dependencies not available
import pyterrier as pt

# PyTerrier 0.11+ auto-starts Java; ensure initialized for earlier versions
if hasattr(pt, 'java') and hasattr(pt.java, 'init'):
    pt.java.init()  # New API
elif not pt.started():
    pt.init()  # Legacy API

import pandas as pd
import numpy as np
from sentence_transformers import CrossEncoder

from .base import BaseRetriever, RetrieverResult


class CrossEncoderReranker(pt.Transformer):
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
        
        # Text loader
        self._corpus_texts = self._load_corpus_texts()
        
        # Build the pipeline
        self.pipeline = self.bm25 >> self._add_text >> self.reranker
        
        print(f"[BM25_MonoT5] Pipeline ready: BM25(k={first_stage_k}) >> CrossEncoder")
    
    def _load_corpus_texts(self) -> Dict[str, str]:
        """Load corpus texts for reranking."""
        import json
        
        corpus_file = os.path.join(self.corpus_path, "corpus.jsonl")
        texts = {}
        
        print(f"[BM25_MonoT5] Loading corpus texts...")
        with open(corpus_file, 'r', encoding='utf-8') as f:
            for line in f:
                doc = json.loads(line)
                doc_id = doc.get("_id", "")
                text = (doc.get("title", "") + " " + doc.get("text", "")).strip()
                texts[doc_id] = text
        
        print(f"[BM25_MonoT5] Loaded {len(texts)} documents")
        return texts
    
    def _add_text(self, df: pd.DataFrame) -> pd.DataFrame:
        """PyTerrier transformer to add text column."""
        df = df.copy()
        df["text"] = df["docno"].apply(lambda x: self._corpus_texts.get(str(x), ""))
        return df
    
    def retrieve(
        self,
        query: str,
        qid: str,
        top_k: int = 100,
        **kwargs
    ) -> RetrieverResult:
        """Retrieve using PyTerrier pipeline."""
        import re
        
        start = time.time()
        
        # Sanitize query
        clean_query = re.sub(r"[^a-zA-Z0-9\s]", " ", query)
        clean_query = re.sub(r"\s+", " ", clean_query).strip()
        
        query_df = pd.DataFrame([{"qid": qid, "query": clean_query}])
        
        # Run pipeline
        results_df = self.pipeline.transform(query_df)
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
        """Batch retrieval using PyTerrier pipeline."""
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
        results_df = self.pipeline.transform(query_df)
        
        # Group by query and build results
        results = {}
        for qid in queries.keys():
            qid_results = results_df[results_df["qid"] == qid].head(top_k)
            
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

"""
Incoming: query, PyTerrier index --- {str, pt.Index}
Processing: BM25 retrieval --- {1 job: sparse retrieval}
Outgoing: ranked results --- {RetrieverResult}

BM25 Retriever via PyTerrier
"""

import re
import time
from typing import Dict, Any, Optional
import pandas as pd

from .base import BaseRetriever, RetrieverResult


def _ensure_pyterrier_init():
    """Lazy PyTerrier initialization to avoid JVM conflicts with pyserini."""
    import pyterrier as pt
    if hasattr(pt, 'java') and hasattr(pt.java, 'init') and not pt.started():
        pt.java.init()
    elif not pt.started():
        pt.init()
    return pt


def sanitize_query(query: str) -> str:
    """Sanitize query for PyTerrier parser - keep only alphanumeric and spaces."""
    # Keep only letters, numbers, and spaces
    query = re.sub(r"[^a-zA-Z0-9\s]", " ", query)
    # Collapse multiple spaces
    query = re.sub(r"\s+", " ", query)
    return query.strip()


class BM25Retriever(BaseRetriever):
    """BM25 sparse retrieval using PyTerrier."""
    
    name = "BM25"
    
    def __init__(self, index_path: str):
        """
        Initialize BM25 retriever.
        
        Args:
            index_path: Path to PyTerrier index directory
        """
        pt = _ensure_pyterrier_init()
        
        self.index = pt.IndexFactory.of(index_path)
        # IMPORTANT: Set num_results in constructor, not per-call
        self.retriever = pt.BatchRetrieve(self.index, wmodel="BM25", num_results=100)
    
    def retrieve(
        self,
        query: str,
        qid: str,
        top_k: int = 100,
        **kwargs
    ) -> RetrieverResult:
        """Retrieve documents using BM25."""
        start = time.time()
        
        # Query as DataFrame with sanitized query
        query_df = pd.DataFrame([{"qid": qid, "query": sanitize_query(query)}])
        
        # Retrieve
        self.retriever.num_results = top_k
        results_df = self.retriever.transform(query_df)
        
        # Convert to result format
        results = []
        for _, row in results_df.iterrows():
            results.append((
                str(row["docno"]),
                float(row["score"]),
                int(row["rank"])
            ))
        
        latency = (time.time() - start) * 1000
        
        return RetrieverResult(
            qid=qid,
            results=results,
            retriever_name=self.name,
            latency_ms=latency,
            metadata={"model": "BM25", "index_path": str(self.index)}
        )
    
    def retrieve_batch(
        self,
        queries: Dict[str, str],
        top_k: int = 100,
        **kwargs
    ) -> Dict[str, RetrieverResult]:
        """Batch retrieval (more efficient for PyTerrier)."""
        import time
        start = time.time()
        
        # Build query DataFrame with sanitized queries
        query_df = pd.DataFrame([
            {"qid": qid, "query": sanitize_query(text)}
            for qid, text in queries.items()
        ])
        
        # Batch retrieve
        self.retriever.num_results = top_k
        results_df = self.retriever.transform(query_df)
        
        # Group by query (OPTIMIZED: O(n) instead of O(n²))
        results = {}
        grouped = results_df.groupby("qid")
        for qid, group in grouped:
            doc_list = []
            for _, row in group.iterrows():
                doc_list.append((
                    str(row["docno"]),
                    float(row["score"]),
                    int(row["rank"])
                ))
            
            results[qid] = RetrieverResult(
                qid=qid,
                results=doc_list,
                retriever_name=self.name,
                latency_ms=0,  # Batch timing
                metadata={"model": "BM25"}
            )
        
        total_time = (time.time() - start) * 1000
        print(f"BM25 batch: {len(queries)} queries in {total_time:.1f}ms")
        
        return results


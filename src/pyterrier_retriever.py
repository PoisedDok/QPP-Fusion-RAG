"""
PyTerrier DR Retriever Operation

Searches a PyTerrier-DR dense index and returns ranked documents.
"""

from __future__ import annotations

import time
from typing import Dict, Any, List, Optional

try:
    import pyterrier as pt  # type: ignore
    import pyterrier_dr as dr  # type: ignore

    if not pt.started():
        pt.init()

    PYT_AVAILABLE = True
except Exception as e:
    PYT_AVAILABLE = False
    _IMPORT_ERROR = str(e)


class PyTerrierRetrieverOperation:
    """Search a PyTerrier-DR dense index."""

    def __init__(self, executor=None):
        if not PYT_AVAILABLE:
            raise RuntimeError("pyterrier or pyterrier_dr not available: " + _IMPORT_ERROR)

    def execute(
        self,
        query: str,
        index_path: str,
        encoder: str | None = None,
        top_k: int = 10,
        batch_size: int = 32,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Search the dense index.

        Args:
            query: Search query text.
            index_path: Path to the PyTerrier-DR index directory.
            encoder: Encoder model name (must match indexing).
            top_k: Number of results to return.
            batch_size: Batch size for encoding queries.

        Returns:
            Dict with documents, scores, query_id, and latency.
        """
        start = time.time()

        if not query or not query.strip():
            return {"error": "Empty query", "documents": [], "scores": [], "count": 0}

        encoder_name = encoder or "sentence-transformers/all-MiniLM-L6-v2"

        # Build retriever
        try:
            dpr_searcher = dr.DPRSearcher(encoder_name, index_path=index_path, batch_size=batch_size)
        except Exception as e:
            return {"error": f"Failed to load index: {str(e)}", "documents": [], "scores": [], "count": 0}

        # Query as DataFrame
        import pandas as pd
        query_df = pd.DataFrame([{"qid": "q1", "query": query}])

        # Retrieve
        try:
            results_df = dpr_searcher.search(query_df)
            results_df = results_df.head(top_k)
        except Exception as e:
            return {"error": f"Search failed: {str(e)}", "documents": [], "scores": [], "count": 0}

        elapsed = time.time() - start

        # Convert to output format
        documents = []
        scores = []
        for _, row in results_df.iterrows():
            documents.append({
                "docno": str(row.get("docno", "")),
                "score": float(row.get("score", 0.0)),
                "rank": int(row.get("rank", 0)),
            })
            scores.append(float(row.get("score", 0.0)))

        return {
            "documents": documents,
            "scores": scores,
            "count": len(documents),
            "query": query,
            "latency_ms": elapsed * 1000,
            "encoder": encoder_name,
        }


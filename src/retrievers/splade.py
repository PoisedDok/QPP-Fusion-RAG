"""
Incoming: query --- {str}
Processing: sparse learned retrieval --- {1 job: Pyserini SPLADE search}
Outgoing: ranked results --- {RetrieverResult}

SPLADE Sparse Learned Retriever (Pyserini)
------------------------------------------
Uses Pyserini's pre-built SPLADE index for BEIR-NQ.
No corpus encoding needed - index downloaded on first use.
"""

import os
import time
from typing import Dict, Optional

from .base import BaseRetriever, RetrieverResult


class SpladeRetriever(BaseRetriever):
    """Sparse learned retrieval using Pyserini pre-built SPLADE index."""
    
    name = "Splade"
    INDEX_NAME = "beir-v1.0.0-nq.splade-pp-ed"
    ENCODER_NAME = "naver/splade-cocondenser-ensembledistil"
    
    def __init__(
        self,
        index_name: Optional[str] = None,
        encoder_name: Optional[str] = None,
        **kwargs  # Accept but ignore legacy params (corpus, cache_dir, etc.)
    ):
        """
        Initialize SPLADE retriever using Pyserini pre-built index.
        
        Args:
            index_name: Pyserini pre-built index name (default: beir-v1.0.0-nq-splade-pp-ed)
            encoder_name: Query encoder model (default: naver/splade-cocondenser-ensembledistil)
            **kwargs: Ignored (for backward compatibility with old interface)
        """
        from pyserini.search.lucene import LuceneImpactSearcher
        
        self.index_name = index_name or self.INDEX_NAME
        self.encoder_name = encoder_name or self.ENCODER_NAME
        
        print(f"[SPLADE] Loading impact index from data folder (git LFS)")
        print(f"[SPLADE] Index: {self.index_name}")
        print(f"[SPLADE] Query encoder: {self.encoder_name}")
        
        # Use index from data folder (tracked in git LFS)
        from pathlib import Path
        project_root = Path(__file__).parent.parent.parent
        dataset = "hotpotqa" if "hotpotqa" in self.index_name else "nq"
        index_dir = project_root / "data" / dataset / "index" / "splade"
        
        if not index_dir.exists():
            raise FileNotFoundError(f"Index not found: {index_dir}")
        
        self.searcher = LuceneImpactSearcher(
            index_dir=str(index_dir),
            query_encoder=self.encoder_name
        )
        
        print(f"[SPLADE] Index loaded. Ready for retrieval.")
    
    def retrieve(
        self,
        query: str,
        qid: str,
        top_k: int = 100,
        **kwargs
    ) -> RetrieverResult:
        """Retrieve documents using SPLADE."""
        start = time.time()
        
        hits = self.searcher.search(query, k=top_k)
        
        results = [
            (hit.docid, float(hit.score), rank + 1)
            for rank, hit in enumerate(hits)
        ]
        
        latency = (time.time() - start) * 1000
        
        return RetrieverResult(
            qid=qid,
            results=results,
            retriever_name=self.name,
            latency_ms=latency,
            metadata={"index": self.index_name, "encoder": self.encoder_name}
        )
    
    def retrieve_batch(
        self,
        queries: Dict[str, str],
        top_k: int = 100,
        **kwargs
    ) -> Dict[str, RetrieverResult]:
        """Batch retrieval using Pyserini."""
        start = time.time()
        
        results = {}
        
        # Pyserini supports batch search
        query_ids = list(queries.keys())
        query_texts = [queries[qid] for qid in query_ids]
        
        # Use batch_search for efficiency
        batch_hits = self.searcher.batch_search(
            queries=query_texts,
            qids=query_ids,
            k=top_k,
            threads=4
        )
        
        for qid in query_ids:
            hits = batch_hits.get(qid, [])
            doc_results = [
                (hit.docid, float(hit.score), rank + 1)
                for rank, hit in enumerate(hits)
            ]
            
            results[qid] = RetrieverResult(
                qid=qid,
                results=doc_results,
                retriever_name=self.name,
                latency_ms=0,
                metadata={"index": self.index_name}
            )
        
        total_time = (time.time() - start) * 1000
        print(f"[SPLADE] Batch: {len(queries)} queries in {total_time:.1f}ms")
        
        return results


# Backward compatibility alias
SpladeRetrieverLegacy = None  # Old implementation removed - use Pyserini

"""
Incoming: query --- {str}
Processing: dense retrieval --- {1 job: Pyserini BGE FAISS search}
Outgoing: ranked results --- {RetrieverResult}

BGE Dense Retriever (Pyserini)
------------------------------
Uses Pyserini's pre-built BGE FAISS index for BEIR-NQ.
No corpus encoding needed - index downloaded on first use (~8GB).
"""

import os
import time
from typing import Dict, Optional

from .base import BaseRetriever, RetrieverResult


class BGERetriever(BaseRetriever):
    """Dense retrieval using Pyserini pre-built BGE FAISS index."""
    
    name = "BGE"
    INDEX_NAME = "beir-v1.0.0-nq.bge-base-en-v1.5"
    ENCODER_NAME = "BAAI/bge-base-en-v1.5"
    
    def __init__(
        self,
        index_name: Optional[str] = None,
        encoder_name: Optional[str] = None,
        **kwargs
    ):
        """
        Initialize BGE retriever using Pyserini pre-built FAISS index.
        
        Args:
            index_name: Pyserini pre-built index name (default: beir-v1.0.0-nq.bge-base-en-v1.5)
            encoder_name: Query encoder model (default: BAAI/bge-base-en-v1.5)
            **kwargs: Ignored (for backward compatibility)
        """
        # Fix for M4 OpenMP threading issue
        os.environ.setdefault('OMP_NUM_THREADS', '1')
        
        from pyserini.search.faiss import FaissSearcher
        
        self.index_name = index_name or self.INDEX_NAME
        self.encoder_name = encoder_name or self.ENCODER_NAME
        
        print(f"[BGE] Loading Pyserini pre-built FAISS index: {self.index_name}")
        print(f"[BGE] Query encoder: {self.encoder_name}")
        print(f"[BGE] NOTE: First run downloads index (~8GB) - subsequent runs use cache")
        
        self.searcher = FaissSearcher.from_prebuilt_index(
            self.index_name,
            self.encoder_name
        )
        
        print(f"[BGE] Index loaded. Ready for retrieval.")
    
    def retrieve(
        self,
        query: str,
        qid: str,
        top_k: int = 100,
        **kwargs
    ) -> RetrieverResult:
        """Retrieve documents using BGE dense embeddings."""
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
        """Batch retrieval using Pyserini FAISS."""
        start = time.time()
        
        results = {}
        
        query_ids = list(queries.keys())
        query_texts = [queries[qid] for qid in query_ids]
        
        # FAISS batch search
        batch_hits = self.searcher.batch_search(
            queries=query_texts,
            q_ids=query_ids,
            k=top_k,
            threads=1  # Single thread for M4 stability
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
        print(f"[BGE] Batch: {len(queries)} queries in {total_time:.1f}ms")
        
        return results

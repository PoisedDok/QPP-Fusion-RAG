"""
Incoming: query --- {str}
Processing: dense retrieval --- {1 job: Pyserini BGE FAISS search}
Outgoing: ranked results --- {RetrieverResult}

BGE Dense Retriever (Pyserini)
------------------------------
Uses Pyserini's pre-built BGE FAISS flat index.
Memory profile:
- NQ: 7.7GB index (2.6M docs)
- HotpotQA: 15GB index (5M docs)
Sequential processing minimizes overhead beyond index size.
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
        # FORCE multi-threading for FAISS (M4 has 10 cores)
        # Override any previous OMP_NUM_THREADS=1 setting from PyTerrier/Java
        os.environ['OMP_NUM_THREADS'] = '10'  # Use all M4 cores
        os.environ['MKL_NUM_THREADS'] = '10'  # Also set MKL if present
        os.environ['OPENBLAS_NUM_THREADS'] = '10'  # Cover all BLAS implementations
        
        from pyserini.search.faiss import FaissSearcher
        
        self.index_name = index_name or self.INDEX_NAME
        self.encoder_name = encoder_name or self.ENCODER_NAME
        
        print(f"[BGE] Loading FAISS index from cache")
        print(f"[BGE] Index: {self.index_name}")
        print(f"[BGE] Query encoder: {self.encoder_name}")
        
        # Detect dataset and use correct index hash
        dataset = "hotpotqa" if "hotpotqa" in self.index_name else "nq"
        cache_dir = os.environ.get("PYSERINI_CACHE", "cache/pyserini")
        
        # Dataset-specific hashes
        index_hashes = {
            "nq": "faiss-flat.beir-v1.0.0-nq.bge-base-en-v1.5.20240107.b738bbbe7ca36532f25189b776d4e153",
            "hotpotqa": "faiss-flat.beir-v1.0.0-hotpotqa.bge-base-en-v1.5.20240107.d2c08665e8cd750bd06ceb7d23897c94"
        }
        
        index_dir = f"{cache_dir}/indexes/{index_hashes[dataset]}"
        
        if not os.path.exists(index_dir):
            raise FileNotFoundError(f"Index not found: {index_dir}")
        
        # FaissSearcher loads both index and encoder model (~2-3GB combined)        
        self.searcher = FaissSearcher(index_dir, self.encoder_name)
        
        # Force FAISS to use multiple threads (environment vars alone may not be enough)
        import faiss
        faiss.omp_set_num_threads(10)  # Explicitly set FAISS threads
        omp_threads = faiss.omp_get_max_threads()        
        print(f"[BGE] Index loaded. Ready for retrieval (FAISS threads: {omp_threads})")
    
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
        """Batch retrieval with multi-threading for optimal M4 performance."""
        import gc
        
        start = time.time()
        results = {}
        
        query_ids = list(queries.keys())
        query_texts = [queries[qid] for qid in query_ids]
        total_queries = len(query_ids)
        
        # Process in chunks to manage memory (embeddings cached, then discarded)
        # Smaller chunks for large corpora (HotpotQA 5M docs)
        chunk_size = 100  # Balance: enough for batching, small enough for memory
        
        print(f"[BGE] Processing {total_queries} queries in chunks of {chunk_size} (multi-threaded)")
        
        for i in range(0, total_queries, chunk_size):
            chunk_ids = query_ids[i:i + chunk_size]
            chunk_texts = query_texts[i:i + chunk_size]            
            # Multi-threaded batch search (FAISS uses all 10 cores)
            batch_hits = self.searcher.batch_search(
                queries=chunk_texts,
                q_ids=chunk_ids,
                k=top_k,
                threads=10  # M4 has 10 cores - use them all
            )            
            for qid in chunk_ids:
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
            
            # Progress reporting
            elapsed = time.time() - start
            queries_done = i + len(chunk_ids)
            qps = queries_done / elapsed
            eta_sec = (total_queries - queries_done) / qps if qps > 0 else 0
            print(f"[BGE] Progress: {queries_done}/{total_queries} queries ({qps:.1f} q/s, ETA: {eta_sec/60:.1f}m)")
            
            # Memory cleanup after each chunk
            gc.collect()
        
        total_time = (time.time() - start) * 1000
        qps = len(queries) / (total_time / 1000)
        print(f"[BGE] Completed: {len(queries)} queries in {total_time/1000:.1f}s ({qps:.1f} q/s)")
        
        return results

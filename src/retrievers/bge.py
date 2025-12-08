"""
Incoming: query --- {str}
Processing: dense retrieval --- {1 job: Pyserini BGE FAISS search}
Outgoing: ranked results --- {RetrieverResult}

BGE Dense Retriever (Pyserini)
------------------------------
Uses Pyserini's pre-built BGE FAISS flat index.
CPU-based with multi-threaded FAISS search.
Supports checkpointing for crash recovery.
"""

import gc
import json
import os
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Configure threading
os.environ['OMP_NUM_THREADS'] = '10'
os.environ['MKL_NUM_THREADS'] = '10'
os.environ['OPENBLAS_NUM_THREADS'] = '10'

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
        threads: int = 10,
        use_mps: bool = True,
        **kwargs
    ):
        """
        Initialize BGE retriever using Pyserini pre-built FAISS index.
        
        Args:
            index_name: Pyserini pre-built index name
            encoder_name: Query encoder model
            threads: Number of threads for FAISS search
            use_mps: Use MPS for query encoding (Apple Silicon)
        """
        from pyserini.search.faiss import FaissSearcher
        from pyserini.encode import AutoQueryEncoder
        import torch
        import faiss
        
        self.index_name = index_name or self.INDEX_NAME
        self.encoder_name = encoder_name or self.ENCODER_NAME
        self.threads = threads
        
        # Determine device for query encoding
        if use_mps and torch.backends.mps.is_available():
            self.device = "mps"
        else:
            self.device = "cpu"
        
        print(f"[BGE] Loading FAISS index...")
        print(f"[BGE] Index: {self.index_name}")
        print(f"[BGE] Query encoder device: {self.device}")
        print(f"[BGE] FAISS threads: {threads}")
        
        # Detect dataset
        dataset = "hotpotqa" if "hotpotqa" in self.index_name else "nq"
        cache_dir = os.environ.get("PYSERINI_CACHE", "cache/pyserini")
        
        index_hashes = {
            "nq": "faiss-flat.beir-v1.0.0-nq.bge-base-en-v1.5.20240107.b738bbbe7ca36532f25189b776d4e153",
            "hotpotqa": "faiss-flat.beir-v1.0.0-hotpotqa.bge-base-en-v1.5.20240107.d2c08665e8cd750bd06ceb7d23897c94"
        }
        
        index_dir = f"{cache_dir}/indexes/{index_hashes[dataset]}"
        
        if not os.path.exists(index_dir):
            raise FileNotFoundError(f"Index not found: {index_dir}")
        
        # Create MPS-accelerated query encoder
        encoder = AutoQueryEncoder(
            encoder_dir=self.encoder_name,
            device=self.device,
            pooling='cls',
            l2_norm=True
        )
        
        self.searcher = FaissSearcher(index_dir, encoder)
        
        faiss.omp_set_num_threads(threads)
        print(f"[BGE] Ready")
    
    def retrieve(self, query: str, qid: str, top_k: int = 100, **kwargs) -> RetrieverResult:
        """Retrieve documents using BGE dense embeddings."""
        start = time.time()
        
        hits = self.searcher.search(query, k=top_k)
        results = [(hit.docid, float(hit.score), rank + 1) for rank, hit in enumerate(hits)]
        
        return RetrieverResult(
            qid=qid, results=results, retriever_name=self.name,
            latency_ms=(time.time() - start) * 1000,
            metadata={"index": self.index_name, "encoder": self.encoder_name}
        )
    
    def _process_mini_batch(
        self,
        queries: List[Tuple[str, str]],
        top_k: int
    ) -> List[RetrieverResult]:
        """Process a mini-batch using FAISS batch search."""
        query_ids = [q[0] for q in queries]
        query_texts = [q[1] for q in queries]
        
        t0 = time.time()
        batch_hits = self.searcher.batch_search(
            queries=query_texts,
            q_ids=query_ids,
            k=top_k,
            threads=self.threads
        )
        print(f"[BGE]     Searched {len(queries)} queries in {time.time()-t0:.1f}s")
        
        results = []
        for qid in query_ids:
            hits = batch_hits.get(qid, [])
            doc_results = [(hit.docid, float(hit.score), rank + 1) for rank, hit in enumerate(hits)]
            results.append(RetrieverResult(
                qid=qid, results=doc_results, retriever_name=self.name,
                latency_ms=0, metadata={"index": self.index_name}
            ))
        
        return results
    
    def retrieve_batch(
        self,
        queries: Dict[str, str],
        top_k: int = 100,
        checkpoint_path: Optional[str] = None,
        mini_batch_size: int = 10,  # Small batches for memory efficiency
        **kwargs
    ) -> Dict[str, RetrieverResult]:
        """
        Batch retrieval with checkpointing.
        
        Args:
            queries: Dict of qid -> query text
            top_k: Docs per query
            checkpoint_path: JSONL file for crash recovery
            mini_batch_size: Queries per mini-batch
        """
        start = time.time()
        n_queries = len(queries)
        
        # Load checkpoint
        completed = {}
        if checkpoint_path and Path(checkpoint_path).exists():
            print(f"[BGE] Loading checkpoint...")
            with open(checkpoint_path, 'r') as f:
                for line in f:
                    data = json.loads(line)
                    qid = data["qid"]
                    completed[qid] = RetrieverResult(
                        qid=qid,
                        results=[(d["docno"], d["score"], d["rank"]) for d in data["results"]],
                        retriever_name=self.name, latency_ms=0, metadata={"index": self.index_name}
                    )
            print(f"[BGE] Resumed: {len(completed)}/{n_queries} queries from checkpoint")
        
        # Filter pending
        pending = [(qid, text) for qid, text in queries.items() if qid not in completed]
        n_pending = len(pending)
        
        if n_pending == 0:
            print(f"[BGE] All {n_queries} queries completed!")
            return completed
        
        print(f"[BGE] Processing {n_pending} queries in batches of {mini_batch_size}")
        
        n_batches = (n_pending + mini_batch_size - 1) // mini_batch_size
        
        for i in range(n_batches):
            batch_start = i * mini_batch_size
            batch_end = min(batch_start + mini_batch_size, n_pending)
            batch = pending[batch_start:batch_end]
            
            t0 = time.time()
            print(f"[BGE] Batch {i+1}/{n_batches}: {len(batch)} queries...")
            
            batch_results = self._process_mini_batch(batch, top_k)
            
            # Save to checkpoint
            if checkpoint_path:
                with open(checkpoint_path, 'a') as f:
                    for r in batch_results:
                        f.write(json.dumps({
                            "qid": r.qid,
                            "results": [{"docno": d[0], "score": d[1], "rank": d[2]} for d in r.results]
                        }) + "\n")
            
            for r in batch_results:
                completed[r.qid] = r
            
            done = len(completed)
            elapsed = time.time() - start
            rate = done / elapsed if elapsed > 0 else 0
            eta = (n_queries - done) / rate if rate > 0 else 0
            print(f"[BGE]   → {done}/{n_queries} done ({time.time()-t0:.1f}s) | ETA: {eta/60:.1f}min")
            
            # Aggressive memory cleanup
            del batch_results
            gc.collect()
            
            # MPS cache cleanup
            try:
                import torch
                if torch.backends.mps.is_available():
                    torch.mps.empty_cache()
            except:
                pass
        
        print(f"[BGE] Complete: {n_queries} queries in {time.time()-start:.1f}s")
        return completed

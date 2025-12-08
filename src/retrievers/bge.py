"""
Incoming: query --- {str}
Processing: dense retrieval --- {1 job: HNSW/FAISS search with BGE encoding}
Outgoing: ranked results --- {RetrieverResult}

BGE Dense Retriever (Direct Index Access)
-----------------------------------------
Uses pre-built indexes loaded directly (bypasses Pyserini FaissSearcher).
Supports HNSW index for 10-100x faster search, falls back to FAISS flat.
Uses sentence-transformers for query encoding with MPS acceleration.

Performance (5.2M vectors, HotPotQA):
- FAISS flat: ~20s per query (brute force)
- HNSW: ~2ms per query (approximate, 99%+ recall)
"""

import gc
import json
import os
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Threading config - allow PyTorch + FAISS/HNSW to coexist
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
os.environ.setdefault('OMP_NUM_THREADS', '8')
os.environ.setdefault('MKL_NUM_THREADS', '8')
os.environ.setdefault('TOKENIZERS_PARALLELISM', 'true')

import numpy as np
import torch
from sentence_transformers import SentenceTransformer

from .base import BaseRetriever, RetrieverResult


def _get_device():
    """Get best available device for encoding."""
    if torch.backends.mps.is_available():
        return "mps"
    elif torch.cuda.is_available():
        return "cuda"
    return "cpu"


class BGERetriever(BaseRetriever):
    """
    Dense retrieval using HNSW (fast) or FAISS flat (slow) index.
    
    Automatically uses HNSW if available (10-100x faster).
    Build HNSW with: python scripts/build_hnsw_index.py --dataset hotpotqa
    """
    
    name = "BGE"
    MODEL_NAME = "BAAI/bge-base-en-v1.5"
    EMBEDDING_DIM = 768
    
    # Pyserini pre-built index hashes
    INDEX_HASHES = {
        "nq": "faiss-flat.beir-v1.0.0-nq.bge-base-en-v1.5.20240107.b738bbbe7ca36532f25189b776d4e153",
        "hotpotqa": "faiss-flat.beir-v1.0.0-hotpotqa.bge-base-en-v1.5.20240107.d2c08665e8cd750bd06ceb7d23897c94"
    }
    
    def __init__(
        self,
        index_name: Optional[str] = None,
        dataset: str = "nq",
        encoder_batch_size: int = 64,
        use_mps: bool = True,
        ef_search: int = 128,  # HNSW search accuracy (higher = more accurate, slower)
        **kwargs
    ):
        """
        Initialize BGE retriever.
        
        Args:
            index_name: Pyserini pre-built index name (extracts dataset)
            dataset: Dataset name ('nq' or 'hotpotqa')
            encoder_batch_size: Batch size for query encoding
            use_mps: Use MPS for query encoding (Apple Silicon)
            ef_search: HNSW ef parameter (64-256 typical, higher = more accurate)
        """
        # Extract dataset from index_name if provided
        if index_name and "hotpotqa" in index_name.lower():
            dataset = "hotpotqa"
        elif index_name and "nq" in index_name.lower():
            dataset = "nq"
        
        self.dataset = dataset
        self.encoder_batch_size = encoder_batch_size
        self.ef_search = ef_search
        self.use_hnsw = False
        self.hnsw_index = None
        self.faiss_index = None
        
        # Determine device for query encoding
        if use_mps and torch.backends.mps.is_available():
            self.device = "mps"
        else:
            self.device = "cpu"
        
        print(f"[BGE] Initializing retriever for {dataset}...")
        print(f"[BGE] Query encoder device: {self.device}")
        
        # Load index and docids
        self._load_index_and_docids()
        
        # Load query encoder
        self._load_encoder()
    
    def _load_index_and_docids(self):
        """Load HNSW (preferred) or FAISS index and document IDs."""
        cache_dir = os.environ.get("PYSERINI_CACHE", "cache/pyserini")
        
        if self.dataset not in self.INDEX_HASHES:
            raise ValueError(f"Unknown dataset: {self.dataset}")
        
        index_hash = self.INDEX_HASHES[self.dataset]
        index_dir = Path(cache_dir) / "indexes" / index_hash
        
        if not index_dir.exists():
            raise FileNotFoundError(f"Index not found: {index_dir}")
        
        # Try HNSW first (much faster)
        hnsw_path = index_dir / "index_hnsw.bin"
        faiss_path = index_dir / "index"
        docid_path = index_dir / "docid"
        
        if hnsw_path.exists():
            print(f"[BGE] Loading HNSW index (fast mode)...")
            t0 = time.time()
            import hnswlib
            
            # Load docids first to get count
            with open(docid_path, 'r') as f:
                self.docids = [line.strip() for line in f]
            
            self.hnsw_index = hnswlib.Index(space='ip', dim=self.EMBEDDING_DIM)
            self.hnsw_index.load_index(str(hnsw_path), max_elements=len(self.docids))
            self.hnsw_index.set_ef(self.ef_search)
            self.hnsw_index.set_num_threads(8)
            self.use_hnsw = True
            
            print(f"[BGE] Loaded HNSW: {len(self.docids):,} vectors ({time.time()-t0:.1f}s)")
            print(f"[BGE] HNSW ef_search={self.ef_search} (higher=more accurate)")
        else:
            print(f"[BGE] HNSW not found, using FAISS flat (slow)...")
            print(f"[BGE] Build HNSW with: python scripts/build_hnsw_index.py --dataset {self.dataset}")
            
            import faiss
            t0 = time.time()
            self.faiss_index = faiss.read_index(str(faiss_path))
            faiss.omp_set_num_threads(8)
            print(f"[BGE] Loaded FAISS: {self.faiss_index.ntotal:,} vectors ({time.time()-t0:.1f}s)")
            
            # Load docids
            with open(docid_path, 'r') as f:
                self.docids = [line.strip() for line in f]
        
        print(f"[BGE] Loaded {len(self.docids):,} document IDs")
    
    def _load_encoder(self):
        """Load BGE query encoder with MPS acceleration."""
        print(f"[BGE] Loading query encoder: {self.MODEL_NAME}...")
        t0 = time.time()
        self.encoder = SentenceTransformer(self.MODEL_NAME, device=self.device)
        print(f"[BGE] Encoder ready on {self.device} ({time.time()-t0:.1f}s)")
    
    def _encode_queries(self, queries: List[str]) -> np.ndarray:
        """Encode queries using BGE encoder with L2 normalization."""
        with torch.no_grad():
            embeddings = self.encoder.encode(
                queries,
                batch_size=self.encoder_batch_size,
                convert_to_numpy=True,
                normalize_embeddings=True,
                show_progress_bar=False
            )
        return embeddings.astype(np.float32)
    
    def retrieve(self, query: str, qid: str, top_k: int = 100, **kwargs) -> RetrieverResult:
        """Retrieve documents using BGE dense embeddings."""
        start = time.time()
        
        # Encode query
        query_emb = self._encode_queries([query])
        
        # Search
        if self.use_hnsw:
            labels, distances = self.hnsw_index.knn_query(query_emb, k=top_k)
            indices = labels[0]
            scores = distances[0]  # HNSW returns distances (higher = more similar for IP)
        else:
            scores, indices = self.faiss_index.search(query_emb, top_k)
            indices = indices[0]
            scores = scores[0]
        
        # Build results
        results = []
        for rank, (idx, score) in enumerate(zip(indices, scores), start=1):
            if 0 <= idx < len(self.docids):
                results.append((self.docids[idx], float(score), rank))
        
        return RetrieverResult(
            qid=qid, results=results, retriever_name=self.name,
            latency_ms=(time.time() - start) * 1000,
            metadata={"dataset": self.dataset, "index_type": "hnsw" if self.use_hnsw else "flat"}
        )
    
    def _process_mini_batch(
        self,
        queries: List[Tuple[str, str]],
        top_k: int
    ) -> List[RetrieverResult]:
        """Process a mini-batch using vectorized search."""
        query_ids = [q[0] for q in queries]
        query_texts = [q[1] for q in queries]
        
        # Encode all queries in batch
        t0 = time.time()
        query_embs = self._encode_queries(query_texts)
        encode_time = time.time() - t0
        
        # Batch search
        t0 = time.time()
        if self.use_hnsw:
            all_labels, all_scores = self.hnsw_index.knn_query(query_embs, k=top_k)
        else:
            all_scores, all_labels = self.faiss_index.search(query_embs, top_k)
        search_time = time.time() - t0
        
        index_type = "HNSW" if self.use_hnsw else "FAISS"
        print(f"[BGE]     Encoded {len(queries)} queries in {encode_time:.2f}s, {index_type} searched in {search_time:.3f}s")
        
        # Build results
        results = []
        for i, qid in enumerate(query_ids):
            doc_results = []
            for rank, (idx, score) in enumerate(zip(all_labels[i], all_scores[i]), start=1):
                if 0 <= idx < len(self.docids):
                    doc_results.append((self.docids[idx], float(score), rank))
            
            results.append(RetrieverResult(
                qid=qid, results=doc_results, retriever_name=self.name,
                latency_ms=0, metadata={"dataset": self.dataset}
            ))
        
        return results
    
    def retrieve_batch(
        self,
        queries: Dict[str, str],
        top_k: int = 100,
        checkpoint_path: Optional[str] = None,
        mini_batch_size: int = 100,
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
                        retriever_name=self.name, latency_ms=0, metadata={"dataset": self.dataset}
                    )
            print(f"[BGE] Resumed: {len(completed)}/{n_queries} queries from checkpoint")
        
        # Filter pending
        pending = [(qid, text) for qid, text in queries.items() if qid not in completed]
        n_pending = len(pending)
        
        if n_pending == 0:
            print(f"[BGE] All {n_queries} queries completed!")
            return completed
        
        index_type = "HNSW" if self.use_hnsw else "FAISS flat"
        print(f"[BGE] Processing {n_pending} queries with {index_type} in batches of {mini_batch_size}")
        
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
            print(f"[BGE]   → {done}/{n_queries} done ({time.time()-t0:.1f}s) | ETA: {eta:.1f}s")
            
            # Memory cleanup
            del batch_results
            gc.collect()
            
            if torch.backends.mps.is_available():
                torch.mps.empty_cache()
        
        print(f"[BGE] Complete: {n_queries} queries in {time.time()-start:.1f}s")
        return completed

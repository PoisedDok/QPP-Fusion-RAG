"""
Incoming: query, corpus --- {str, Dict}
Processing: dense retrieval --- {1 job: embedding + similarity}
Outgoing: ranked results --- {RetrieverResult}

TCT-ColBERT Dense Retriever
---------------------------
Optimized for Mac M4 16GB:
- Uses MPS acceleration
- fp16 embeddings (half memory)
- Chunked encoding with disk caching
- Memory-efficient FAISS index
"""

import os
import gc
import time
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Optional

from .base import BaseRetriever, RetrieverResult


class TCTColBERTRetriever(BaseRetriever):
    """Dense retrieval using TCT-ColBERT with memory optimizations."""
    
    name = "TCT-ColBERT"
    MODEL_NAME = "castorini/tct_colbert-v2-hnp-msmarco"
    EMBEDDING_DIM = 768
    
    def __init__(
        self,
        corpus: Dict[str, Dict[str, str]],
        cache_dir: Optional[str] = None,
        batch_size: int = 64,
        use_fp16: bool = True,
        use_faiss: bool = True
    ):
        """
        Initialize TCT-ColBERT retriever with memory optimizations.
        
        Args:
            corpus: {doc_id: {text, title}}
            cache_dir: Directory to cache embeddings
            batch_size: Batch size for encoding
            use_fp16: Use float16 to halve memory
            use_faiss: Use FAISS for efficient search
        """
        import torch
        from sentence_transformers import SentenceTransformer
        
        self.corpus = corpus
        self.batch_size = batch_size
        self.use_fp16 = use_fp16
        self.use_faiss = use_faiss
        self.doc_ids = list(corpus.keys())
        self.cache_dir = Path(cache_dir) if cache_dir else Path("data/nq/cache")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Device setup - prefer MPS on Mac
        if torch.backends.mps.is_available():
            self.device = "mps"
        elif torch.cuda.is_available():
            self.device = "cuda"
        else:
            self.device = "cpu"
        
        print(f"[TCT-ColBERT] Device: {self.device}, fp16: {use_fp16}")
        print(f"[TCT-ColBERT] Loading {self.MODEL_NAME}...")
        
        self.model = SentenceTransformer(self.MODEL_NAME, device=self.device)
        
        # Load or compute embeddings
        self._load_or_compute_index()
    
    def _get_cache_path(self) -> Path:
        """Get cache file path."""
        n_docs = len(self.doc_ids)
        dtype = "fp16" if self.use_fp16 else "fp32"
        return self.cache_dir / f"tct_colbert_{n_docs}_{dtype}.npy"
    
    def _load_or_compute_index(self):
        """Load cached embeddings or compute new ones."""
        cache_path = self._get_cache_path()
        ids_path = self.cache_dir / f"tct_colbert_{len(self.doc_ids)}_ids.npy"
        
        if cache_path.exists() and ids_path.exists():
            print(f"[TCT-ColBERT] Loading cached embeddings from {cache_path}")
            self.doc_embeddings = np.load(str(cache_path))
            cached_ids = np.load(str(ids_path), allow_pickle=True)
            
            # Verify IDs match
            if list(cached_ids) == self.doc_ids:
                print(f"[TCT-ColBERT] Loaded {len(self.doc_embeddings)} embeddings")
                self._build_index()
                return
            else:
                print("[TCT-ColBERT] Cache IDs mismatch, recomputing...")
        
        # Compute embeddings
        self._encode_corpus_chunked()
        
        # Save cache
        np.save(str(cache_path), self.doc_embeddings)
        np.save(str(ids_path), np.array(self.doc_ids, dtype=object))
        print(f"[TCT-ColBERT] Saved embeddings to {cache_path}")
        
        self._build_index()
    
    def _encode_corpus_chunked(self):
        """Encode corpus in memory-efficient chunks."""
        import torch
        
        print(f"[TCT-ColBERT] Encoding {len(self.doc_ids)} documents...")
        
        dtype = np.float16 if self.use_fp16 else np.float32
        n_docs = len(self.doc_ids)
        
        # Pre-allocate array
        self.doc_embeddings = np.zeros((n_docs, self.EMBEDDING_DIM), dtype=dtype)
        
        chunk_size = self.batch_size * 8  # Process in larger chunks
        
        for start_idx in range(0, n_docs, chunk_size):
            end_idx = min(start_idx + chunk_size, n_docs)
            
            # Get texts for this chunk
            chunk_ids = self.doc_ids[start_idx:end_idx]
            chunk_texts = [
                (self.corpus[d].get("title", "") + " " + self.corpus[d].get("text", "")).strip()
                for d in chunk_ids
            ]
            
            # Encode
            with torch.no_grad():
                embeddings = self.model.encode(
                    chunk_texts,
                    batch_size=self.batch_size,
                    convert_to_numpy=True,
                    normalize_embeddings=True,
                    show_progress_bar=False
                )
            
            # Store as fp16 if requested
            if self.use_fp16:
                embeddings = embeddings.astype(np.float16)
            
            self.doc_embeddings[start_idx:end_idx] = embeddings
            
            # Progress
            progress = (end_idx / n_docs) * 100
            print(f"  [{progress:5.1f}%] Encoded {end_idx}/{n_docs} documents")
            
            # Memory cleanup
            del embeddings, chunk_texts
            gc.collect()
        
        print(f"[TCT-ColBERT] Encoding complete. Shape: {self.doc_embeddings.shape}")
    
    def _build_index(self):
        """Build FAISS index for efficient search."""
        if not self.use_faiss:
            self.index = None
            return
        
        try:
            import faiss
            
            print("[TCT-ColBERT] Building FAISS index...")
            
            # Convert to float32 for FAISS (required)
            embeddings_f32 = self.doc_embeddings.astype(np.float32)
            
            # Use IndexFlatIP for cosine similarity (embeddings are normalized)
            self.index = faiss.IndexFlatIP(self.EMBEDDING_DIM)
            self.index.add(embeddings_f32)
            
            print(f"[TCT-ColBERT] FAISS index built: {self.index.ntotal} vectors")
            
            # Free the float32 copy
            del embeddings_f32
            gc.collect()
            
        except ImportError:
            print("[TCT-ColBERT] FAISS not available, using numpy")
            self.index = None
    
    def retrieve(
        self,
        query: str,
        qid: str,
        top_k: int = 100,
        **kwargs
    ) -> RetrieverResult:
        """Retrieve documents using dense similarity."""
        import torch
        start = time.time()
        
        # Encode query
        with torch.no_grad():
            query_emb = self.model.encode(
                [query],
                convert_to_numpy=True,
                normalize_embeddings=True
            )[0].astype(np.float32)
        
        # Search
        if self.index is not None:
            # FAISS search
            scores, indices = self.index.search(query_emb.reshape(1, -1), top_k)
            scores = scores[0]
            indices = indices[0]
        else:
            # Numpy fallback
            doc_emb_f32 = self.doc_embeddings.astype(np.float32)
            scores = np.dot(doc_emb_f32, query_emb)
            indices = np.argsort(scores)[::-1][:top_k]
            scores = scores[indices]
        
        results = [
            (self.doc_ids[idx], float(scores[i]), i + 1)
            for i, idx in enumerate(indices)
        ]
        
        latency = (time.time() - start) * 1000
        
        return RetrieverResult(
            qid=qid,
            results=results,
            retriever_name=self.name,
            latency_ms=latency,
            metadata={"model": self.MODEL_NAME, "device": self.device}
        )
    
    def retrieve_batch(
        self,
        queries: Dict[str, str],
        top_k: int = 100,
        **kwargs
    ) -> Dict[str, RetrieverResult]:
        """Batch retrieval with vectorized similarity."""
        import torch
        start = time.time()
        
        query_ids = list(queries.keys())
        query_texts = [queries[q] for q in query_ids]
        
        # Encode all queries
        with torch.no_grad():
            query_embs = self.model.encode(
                query_texts,
                batch_size=self.batch_size,
                convert_to_numpy=True,
                normalize_embeddings=True,
                show_progress_bar=len(query_texts) > 100
            ).astype(np.float32)
        
        results = {}
        
        if self.index is not None:
            # FAISS batch search
            all_scores, all_indices = self.index.search(query_embs, top_k)
            
            for i, qid in enumerate(query_ids):
                doc_results = [
                    (self.doc_ids[idx], float(all_scores[i, j]), j + 1)
                    for j, idx in enumerate(all_indices[i])
                    if idx >= 0  # FAISS returns -1 for empty
                ]
                results[qid] = RetrieverResult(
                    qid=qid,
                    results=doc_results,
                    retriever_name=self.name,
                    latency_ms=0,
                    metadata={"model": self.MODEL_NAME}
                )
        else:
            # Numpy fallback (memory intensive for large corpus)
            doc_emb_f32 = self.doc_embeddings.astype(np.float32)
            all_scores = np.dot(doc_emb_f32, query_embs.T)
            
            for i, qid in enumerate(query_ids):
                scores = all_scores[:, i]
                top_indices = np.argsort(scores)[::-1][:top_k]
                
                doc_results = [
                    (self.doc_ids[idx], float(scores[idx]), rank + 1)
                    for rank, idx in enumerate(top_indices)
                ]
                results[qid] = RetrieverResult(
                    qid=qid,
                    results=doc_results,
                    retriever_name=self.name,
                    latency_ms=0,
                    metadata={"model": self.MODEL_NAME}
                )
        
        total_time = (time.time() - start) * 1000
        print(f"[TCT-ColBERT] Batch: {len(queries)} queries in {total_time:.1f}ms")
        
        return results

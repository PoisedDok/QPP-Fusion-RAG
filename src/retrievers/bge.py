"""
Incoming: query --- {str}
Processing: dense retrieval --- {1 job: HNSW search with BGE encoding}
Outgoing: ranked results --- {RetrieverResult}

BGE Dense Retriever using segmented HNSW index.
- Loads pre-built HNSW segments for memory-efficient search
- Uses sentence-transformers for query encoding (MPS/CUDA/CPU)
- ~6ms search latency on 5.2M vectors

Build index: python scripts/01_index.py --dataset hotpotqa --hnsw
"""
import gc
import json
import os
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import hnswlib
import numpy as np
import torch
from sentence_transformers import SentenceTransformer

# Import config first - sets up environment
from src.config import config, get_device

from .base import BaseRetriever, RetrieverResult


class BGERetriever(BaseRetriever):
    """
    Dense retrieval using segmented HNSW index with BGE embeddings.
    
    Build index first:
        python scripts/01_index.py --dataset hotpotqa --hnsw
    """
    
    name = "BGE"
    
    def __init__(
        self,
        index_name: Optional[str] = None,
        dataset: str = "nq",
        encoder_batch_size: Optional[int] = None,
        use_mps: bool = True,
        ef_search: Optional[int] = None,
        **kwargs
    ):
        """
        Initialize BGE retriever.
        
        Args:
            index_name: Pyserini index name (extracts dataset automatically)
            dataset: Dataset name ('nq' or 'hotpotqa')
            encoder_batch_size: Batch size for query encoding (from config if None)
            use_mps: Use MPS for encoding (Apple Silicon)
            ef_search: HNSW search accuracy (from config if None)
        """
        # Get config values
        self.model_name = config.models.bge.name
        self.embedding_dim = config.models.bge.embedding_dim
        
        # Parse dataset from index_name if provided
        if index_name:
            for ds in config.datasets.supported:
                if ds in index_name.lower():
                    dataset = ds
                    break
        
        self.dataset = dataset
        self.encoder_batch_size = encoder_batch_size or config.processing.batch_sizes.bge_encoding
        self.ef_search = ef_search or config.indexes.hnsw.ef_search
        self.hnsw_segments: List[Tuple[hnswlib.Index, int]] = []
        
        # Device selection
        self.device = get_device() if use_mps else "cpu"
        
        print(f"[BGE] Initializing for {dataset}...")
        print(f"[BGE] Encoder device: {self.device}")
        
        self._load_index()
        self._load_encoder()
    
    def _load_index(self):
        """Load HNSW segments and document IDs."""
        cache_dir = Path(os.environ.get("PYSERINI_CACHE", str(config.cache_root / "pyserini")))
        
        # Get index hash from config
        index_hash = config.get_index_hash("bge", self.dataset)
        index_dir = cache_dir / "indexes" / index_hash
        
        if not index_dir.exists():
            raise FileNotFoundError(f"Index not found: {index_dir}")
        
        docid_path = index_dir / "docid"
        metadata_path = index_dir / "hnsw_segments_meta.json"
        
        # Load docids
        with open(docid_path, 'r') as f:
            self.docids = [line.strip() for line in f]
        
        # Load HNSW segments
        if not metadata_path.exists():
            raise FileNotFoundError(
                f"HNSW index not found. Build with:\n"
                f"  python scripts/01_index.py --dataset {self.dataset} --hnsw"
            )
        
        print(f"[BGE] Loading HNSW segments...")
        t0 = time.time()
        
        with open(metadata_path, 'r') as f:
            meta = json.load(f)
        
        for seg_info in meta["segments"]:
            seg_path = index_dir / seg_info["path"]
            if not seg_path.exists():
                raise FileNotFoundError(f"Segment missing: {seg_path}")
            
            seg_size = seg_info["end_global_id"] - seg_info["start_global_id"]
            hnsw = hnswlib.Index(space='ip', dim=self.embedding_dim)
            hnsw.load_index(str(seg_path), max_elements=seg_size)
            hnsw.set_ef(self.ef_search)
            hnsw.set_num_threads(4)
            
            self.hnsw_segments.append((hnsw, seg_info["start_global_id"]))
        
        print(f"[BGE] Loaded {len(self.hnsw_segments)} segments ({time.time()-t0:.1f}s)")
        print(f"[BGE] Total vectors: {len(self.docids):,}")
        print(f"[BGE] ef_search={self.ef_search}")
    
    def _load_encoder(self):
        """Load BGE query encoder."""
        print(f"[BGE] Loading encoder: {self.model_name}...")
        t0 = time.time()
        self.encoder = SentenceTransformer(self.model_name, device=self.device)
        print(f"[BGE] Encoder ready ({time.time()-t0:.1f}s)")
    
    def _encode_queries(self, queries: List[str]) -> np.ndarray:
        """Encode queries with L2 normalization."""
        with torch.no_grad():
            embeddings = self.encoder.encode(
                queries,
                batch_size=self.encoder_batch_size,
                convert_to_numpy=True,
                normalize_embeddings=True,
                show_progress_bar=False
            )
        return embeddings.astype(np.float32)
    
    def _search_segments(self, query_emb: np.ndarray, top_k: int) -> Tuple[np.ndarray, np.ndarray]:
        """Search all segments and merge results."""
        all_ids = []
        all_scores = []
        
        for hnsw, start_id in self.hnsw_segments:
            labels, distances = hnsw.knn_query(query_emb, k=top_k)
            all_ids.extend(labels[0] + start_id)
            all_scores.extend(distances[0])
        
        all_scores = np.array(all_scores)
        all_ids = np.array(all_ids)
        sorted_idx = np.argsort(-all_scores)[:top_k]
        
        return all_ids[sorted_idx], all_scores[sorted_idx]
    
    def _search_segments_batch(self, query_embs: np.ndarray, top_k: int) -> Tuple[np.ndarray, np.ndarray]:
        """Batch search all segments and merge results."""
        n_queries = query_embs.shape[0]
        segment_results = []
        
        for hnsw, start_id in self.hnsw_segments:
            labels, distances = hnsw.knn_query(query_embs, k=top_k)
            segment_results.append((labels + start_id, distances))
        
        merged_ids = np.zeros((n_queries, top_k), dtype=np.int64)
        merged_scores = np.zeros((n_queries, top_k), dtype=np.float32)
        
        for q_idx in range(n_queries):
            q_ids = np.concatenate([r[0][q_idx] for r in segment_results])
            q_scores = np.concatenate([r[1][q_idx] for r in segment_results])
            sorted_idx = np.argsort(-q_scores)[:top_k]
            merged_ids[q_idx] = q_ids[sorted_idx]
            merged_scores[q_idx] = q_scores[sorted_idx]
        
        return merged_ids, merged_scores
    
    def retrieve(self, query: str, qid: str, top_k: int = None, **kwargs) -> RetrieverResult:
        """Retrieve documents for a single query."""
        top_k = top_k or config.processing.retrieval.top_k
        start = time.time()
        
        query_emb = self._encode_queries([query])
        indices, scores = self._search_segments(query_emb, top_k)
        
        results = [
            (self.docids[idx], float(score), rank)
            for rank, (idx, score) in enumerate(zip(indices, scores), start=1)
            if 0 <= idx < len(self.docids)
        ]
        
        return RetrieverResult(
            qid=qid,
            results=results,
            retriever_name=self.name,
            latency_ms=(time.time() - start) * 1000,
            metadata={"dataset": self.dataset, "index_type": "hnsw-segmented"}
        )
    
    def retrieve_batch(
        self,
        queries: Dict[str, str],
        top_k: int = None,
        checkpoint_path: Optional[str] = None,
        mini_batch_size: Optional[int] = None,
        **kwargs
    ) -> Dict[str, RetrieverResult]:
        """
        Batch retrieval with optional checkpointing.
        
        Args:
            queries: Dict of qid -> query text
            top_k: Documents per query
            checkpoint_path: JSONL file for crash recovery
            mini_batch_size: Queries per mini-batch
        """
        top_k = top_k or config.processing.retrieval.top_k
        mini_batch_size = mini_batch_size or config.processing.batch_sizes.bge_mini
        
        start = time.time()
        n_queries = len(queries)
        
        # Load checkpoint
        completed = {}
        if checkpoint_path and Path(checkpoint_path).exists():
            print(f"[BGE] Loading checkpoint...")
            with open(checkpoint_path, 'r') as f:
                for line in f:
                    data = json.loads(line)
                    completed[data["qid"]] = RetrieverResult(
                        qid=data["qid"],
                        results=[(d["docno"], d["score"], d["rank"]) for d in data["results"]],
                        retriever_name=self.name,
                        latency_ms=0,
                        metadata={"dataset": self.dataset}
                    )
            print(f"[BGE] Resumed: {len(completed)}/{n_queries}")
        
        pending = [(qid, text) for qid, text in queries.items() if qid not in completed]
        
        if not pending:
            print(f"[BGE] All {n_queries} queries done!")
            return completed
        
        n_batches = (len(pending) + mini_batch_size - 1) // mini_batch_size
        print(f"[BGE] Processing {len(pending)} queries in {n_batches} batches")
        
        for i in range(n_batches):
            batch = pending[i * mini_batch_size:(i + 1) * mini_batch_size]
            qids = [q[0] for q in batch]
            texts = [q[1] for q in batch]
            
            t0 = time.time()
            query_embs = self._encode_queries(texts)
            encode_time = time.time() - t0
            
            t0 = time.time()
            all_ids, all_scores = self._search_segments_batch(query_embs, top_k)
            search_time = time.time() - t0
            
            print(f"[BGE] Batch {i+1}/{n_batches}: encode={encode_time:.2f}s, search={search_time:.3f}s")
            
            batch_results = []
            for j, qid in enumerate(qids):
                results = [
                    (self.docids[idx], float(score), rank)
                    for rank, (idx, score) in enumerate(zip(all_ids[j], all_scores[j]), start=1)
                    if 0 <= idx < len(self.docids)
                ]
                batch_results.append(RetrieverResult(
                    qid=qid,
                    results=results,
                    retriever_name=self.name,
                    latency_ms=0,
                    metadata={"dataset": self.dataset}
                ))
            
            if checkpoint_path:
                with open(checkpoint_path, 'a') as f:
                    for r in batch_results:
                        f.write(json.dumps({
                            "qid": r.qid,
                            "results": [{"docno": d[0], "score": d[1], "rank": d[2]} for d in r.results]
                        }) + "\n")
            
            for r in batch_results:
                completed[r.qid] = r
            
            elapsed = time.time() - start
            rate = len(completed) / elapsed
            eta = (n_queries - len(completed)) / rate if rate > 0 else 0
            print(f"[BGE]   {len(completed)}/{n_queries} done | ETA: {eta:.0f}s")
            
            del batch_results, query_embs
            gc.collect()
            if torch.backends.mps.is_available():
                torch.mps.empty_cache()
        
        print(f"[BGE] Complete: {n_queries} queries in {time.time()-start:.1f}s")
        return completed

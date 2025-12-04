"""
Incoming: query, index, corpus_path --- {str, pt.Index, str}
Processing: two-stage retrieval --- {2 jobs: BM25 recall, TCT rerank}
Outgoing: ranked results --- {RetrieverResult}

BM25 >> TCT-ColBERT Hybrid Retriever
OPTIMIZED V2: Macro-batch processing, doc embedding caching, ~10x faster
"""

import re
import os
import json
import time
import numpy as np
from typing import Dict, List, Tuple

from .base import BaseRetriever, RetrieverResult


def sanitize_query(query: str) -> str:
    """Sanitize query for PyTerrier parser."""
    query = re.sub(r"[^a-zA-Z0-9\s]", " ", query)
    query = re.sub(r"\s+", " ", query)
    return query.strip()


class BM25TCTRetriever(BaseRetriever):
    """Two-stage: BM25 first-stage + TCT-ColBERT reranking.
    
    OPTIMIZED V2:
    - Macro-batch processing (50 queries at a time)
    - Document embeddings computed ONCE per unique doc per batch
    - Query embeddings batched
    - Minimal memory footprint
    """
    
    name = "BM25_TCT"
    TCT_MODEL = "castorini/tct_colbert-v2-hnp-msmarco"
    MACRO_BATCH_SIZE = 50  # Queries per batch
    
    def __init__(
        self,
        index_path: str,
        corpus_path: str,
        first_stage_k: int = 100,
        batch_size: int = 128  # Embedding batch size
    ):
        import pyterrier as pt
        from sentence_transformers import SentenceTransformer
        
        if not pt.started():
            pt.init()
        
        self.corpus_path = corpus_path
        self.corpus_file = os.path.join(corpus_path, "corpus.jsonl")
        self.first_stage_k = first_stage_k
        self.batch_size = batch_size
        
        # Build doc_id -> offset map
        self._doc_offsets = self._build_offset_index()
        
        # BM25
        self.index = pt.IndexFactory.of(index_path)
        self.bm25 = pt.BatchRetrieve(self.index, wmodel="BM25", num_results=first_stage_k)
        
        # TCT-ColBERT on MPS
        print(f"Loading {self.TCT_MODEL}...")
        self.tct_model = SentenceTransformer(self.TCT_MODEL, device="mps")
    
    def _build_offset_index(self) -> Dict[str, int]:
        """Build doc_id -> byte offset map."""
        print(f"Building corpus offset index...")
        offsets = {}
        with open(self.corpus_file, 'r', encoding='utf-8') as f:
            offset = 0
            for i, line in enumerate(f):
                doc = json.loads(line)
                offsets[doc.get("_id", str(i))] = offset
                offset += len(line.encode('utf-8'))
        print(f"  Indexed {len(offsets)} documents")
        return offsets
    
    def _load_docs(self, doc_ids: List[str]) -> Dict[str, str]:
        """Load specific docs by ID, return {doc_id: text}."""
        ids_with_offset = [(d, self._doc_offsets.get(d, -1)) for d in doc_ids]
        ids_with_offset = [(d, o) for d, o in ids_with_offset if o >= 0]
        ids_with_offset.sort(key=lambda x: x[1])
        
        result = {}
        with open(self.corpus_file, 'r', encoding='utf-8') as f:
            for doc_id, offset in ids_with_offset:
                f.seek(offset)
                doc = json.loads(f.readline())
                result[doc_id] = (doc.get("title", "") + " " + doc.get("text", "")).strip()
        
        return result
    
    def retrieve(self, query: str, qid: str, top_k: int = 100, **kwargs) -> RetrieverResult:
        """Single query retrieval."""
        import pandas as pd
        start = time.time()
        
        query_df = pd.DataFrame([{"qid": qid, "query": sanitize_query(query)}])
        bm25_results = self.bm25.transform(query_df)
        candidates = [str(row["docno"]) for _, row in bm25_results.iterrows()]
        
        if not candidates:
            return RetrieverResult(qid=qid, results=[], retriever_name=self.name,
                                   latency_ms=(time.time()-start)*1000, metadata={})
        
        docs = self._load_docs(candidates)
        valid_ids = [d for d in candidates if d in docs]
        doc_texts = [docs[d] for d in valid_ids]
        
        if not doc_texts:
            return RetrieverResult(qid=qid, results=[], retriever_name=self.name,
                                   latency_ms=(time.time()-start)*1000, metadata={})
        
        query_emb = self.tct_model.encode([query], normalize_embeddings=True)[0]
        doc_embs = self.tct_model.encode(doc_texts, normalize_embeddings=True, batch_size=self.batch_size)
        
        scores = np.dot(doc_embs, query_emb)
        ranked = np.argsort(scores)[::-1][:top_k]
        
        results = [(valid_ids[i], float(scores[i]), r+1) for r, i in enumerate(ranked)]
        return RetrieverResult(qid=qid, results=results, retriever_name=self.name,
                               latency_ms=(time.time()-start)*1000, metadata={})
    
    def retrieve_batch(self, queries: Dict[str, str], top_k: int = 100, **kwargs) -> Dict[str, RetrieverResult]:
        """Optimized batch retrieval with macro-batching."""
        import pandas as pd
        import gc
        
        start = time.time()
        n_queries = len(queries)
        print(f"BM25>>TCT: {n_queries} queries, first_stage_k={self.first_stage_k}, macro_batch={self.MACRO_BATCH_SIZE}")
        
        # Stage 1: BM25 for ALL queries at once
        print(f"  Stage 1: BM25 retrieval...")
        bm25_start = time.time()
        query_df = pd.DataFrame([
            {"qid": qid, "query": sanitize_query(text)}
            for qid, text in queries.items()
        ])
        bm25_results = self.bm25.transform(query_df)
        
        # Group by query
        bm25_by_query: Dict[str, List[str]] = {}
        for _, row in bm25_results.iterrows():
            qid = str(row["qid"])
            bm25_by_query.setdefault(qid, []).append(str(row["docno"]))
        
        print(f"  BM25 done in {time.time() - bm25_start:.1f}s")
        del bm25_results, query_df
        gc.collect()
        
        # Stage 2: Process in macro-batches
        print(f"  Stage 2: TCT-ColBERT reranking (macro-batch={self.MACRO_BATCH_SIZE})...")
        results = {}
        query_items = list(queries.items())
        n_batches = (n_queries + self.MACRO_BATCH_SIZE - 1) // self.MACRO_BATCH_SIZE
        
        for batch_idx in range(n_batches):
            batch_start = time.time()
            start_i = batch_idx * self.MACRO_BATCH_SIZE
            end_i = min(start_i + self.MACRO_BATCH_SIZE, n_queries)
            batch_queries = query_items[start_i:end_i]
            
            # Collect all unique doc_ids needed for this batch
            all_doc_ids = set()
            for qid, _ in batch_queries:
                all_doc_ids.update(bm25_by_query.get(qid, []))
            
            all_doc_ids_list = list(all_doc_ids)
            
            # Load docs for this batch only
            docs = self._load_docs(all_doc_ids_list)
            
            # Build doc texts list (only valid docs)
            valid_doc_ids = [d for d in all_doc_ids_list if d in docs]
            doc_texts = [docs[d] for d in valid_doc_ids]
            doc_id_to_idx = {d: i for i, d in enumerate(valid_doc_ids)}
            
            if not doc_texts:
                for qid, _ in batch_queries:
                    results[qid] = RetrieverResult(qid=qid, results=[], 
                                                   retriever_name=self.name, latency_ms=0, metadata={})
                continue
            
            # Encode all docs for this batch ONCE
            doc_embs = self.tct_model.encode(doc_texts, normalize_embeddings=True, 
                                              batch_size=self.batch_size, show_progress_bar=False)
            
            # Encode all queries in this batch at once
            batch_query_texts = [q for _, q in batch_queries]
            query_embs = self.tct_model.encode(batch_query_texts, normalize_embeddings=True,
                                                batch_size=self.batch_size, show_progress_bar=False)
            
            # Score each query against its candidates
            for i, (qid, _) in enumerate(batch_queries):
                candidates = bm25_by_query.get(qid, [])
                valid_cands = [d for d in candidates if d in doc_id_to_idx]
                
                if not valid_cands:
                    results[qid] = RetrieverResult(qid=qid, results=[], 
                                                   retriever_name=self.name, latency_ms=0, metadata={})
                    continue
                
                # Get embeddings for this query's candidates
                cand_indices = [doc_id_to_idx[d] for d in valid_cands]
                cand_embs = doc_embs[cand_indices]
                
                # Score
                scores = np.dot(cand_embs, query_embs[i])
                ranked = np.argsort(scores)[::-1][:top_k]
                
                top_results = [(valid_cands[j], float(scores[j]), rank+1) for rank, j in enumerate(ranked)]
                
                results[qid] = RetrieverResult(
                    qid=qid, results=top_results, retriever_name=self.name,
                    latency_ms=0, metadata={"stage1_count": len(candidates)}
                )
            
            # Memory cleanup
            del docs, doc_texts, doc_embs, query_embs, doc_id_to_idx
            gc.collect()
            
            # Progress
            elapsed = time.time() - start
            rate = end_i / elapsed
            eta = (n_queries - end_i) / rate if rate > 0 else 0
            batch_time = time.time() - batch_start
            print(f"    Batch {batch_idx+1}/{n_batches}: {end_i}/{n_queries} queries, "
                  f"{rate:.1f} q/s, batch: {batch_time:.1f}s, ETA: {eta/60:.1f}min")
        
        total_time = time.time() - start
        print(f"BM25>>TCT complete: {n_queries} queries in {total_time:.1f}s ({n_queries/total_time:.1f} q/s)")
        
        return results

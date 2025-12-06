"""
Incoming: query, index, corpus_path --- {str, pt.Index, str}
Processing: two-stage retrieval --- {PyTerrier pipeline: BM25 >> TCT-ColBERT}
Outgoing: ranked results --- {RetrieverResult}

BM25 >> TCT-ColBERT Hybrid Retriever (PyTerrier Pipeline)
---------------------------------------------------------
Uses PyTerrier pipeline composition for two-stage retrieval:
1. BM25 first-stage recall
2. TCT-ColBERT dense reranking

STRICT: Uses pyterrier_dr for dense retrieval. No manual fallbacks.
"""

import os
import time
from typing import Dict

from .base import BaseRetriever, RetrieverResult


def _ensure_pyterrier_init():
    """Lazy PyTerrier initialization to avoid JVM conflicts with pyserini."""
    import pyterrier as pt
    if hasattr(pt, 'java') and hasattr(pt.java, 'init') and not pt.started():
        pt.java.init()
    elif not pt.started():
        pt.init()
    return pt


class BM25TCTRetriever(BaseRetriever):
    """
    Two-stage retrieval using PyTerrier pipeline composition.
    
    Pipeline: BM25 >> text_loader >> TCT-ColBERT
    
    Uses pyterrier_dr for dense reranking.
    """
    
    name = "BM25_TCT"
    TCT_MODEL = "castorini/tct_colbert-v2-hnp-msmarco"
    
    def __init__(
        self,
        index_path: str,
        corpus_path: str,
        first_stage_k: int = 100,
        batch_size: int = 128
    ):
        """
        Initialize BM25>>TCT pipeline.
        
        Args:
            index_path: Path to PyTerrier index
            corpus_path: Path to BEIR corpus directory
            first_stage_k: Number of candidates from BM25
            batch_size: Batch size for dense encoding
        """
        import pyterrier_dr as dr
        
        pt = _ensure_pyterrier_init()
        
        self.corpus_path = corpus_path
        self.first_stage_k = first_stage_k
        self.batch_size = batch_size
        
        # BM25 first stage
        self.index = pt.IndexFactory.of(index_path)
        self.bm25 = pt.BatchRetrieve(
            self.index, 
            wmodel="BM25", 
            num_results=first_stage_k,
            metadata=["docno"]
        )
        
        # TCT-ColBERT reranker via pyterrier_dr
        print(f"[BM25_TCT] Loading {self.TCT_MODEL}...")
        
        # Use pyterrier_dr's TctColBert for reranking
        # This creates a proper PyTerrier transformer
        self.tct_reranker = dr.TctColBert.from_pretrained(
            self.TCT_MODEL,
            batch_size=batch_size,
            verbose=False
        )
        
        # Text loader for getting document content
        self._corpus_texts = self._load_corpus_texts()
        
        # Pipeline is run manually (no >> operator) to avoid JVM init conflicts
        print(f"[BM25_TCT] Pipeline ready: BM25(k={first_stage_k}) >> TCT-ColBERT")
    
    def _load_corpus_texts(self) -> Dict[str, str]:
        """Load corpus texts for reranking."""
        import json
        
        corpus_file = os.path.join(self.corpus_path, "corpus.jsonl")
        texts = {}
        
        print(f"[BM25_TCT] Loading corpus texts...")
        with open(corpus_file, 'r', encoding='utf-8') as f:
            for line in f:
                doc = json.loads(line)
                doc_id = doc.get("_id", "")
                text = (doc.get("title", "") + " " + doc.get("text", "")).strip()
                texts[doc_id] = text
        
        print(f"[BM25_TCT] Loaded {len(texts)} documents")
        return texts
    
    def _add_text(self, df):
        """Add text column from corpus."""
        import pandas as pd
        
        df = df.copy()
        df["text"] = df["docno"].apply(lambda x: self._corpus_texts.get(str(x), ""))
        return df
    
    def _run_pipeline(self, query_df):
        """Run the manual BM25 >> text >> TCT-ColBERT pipeline."""
        # Step 1: BM25 retrieval
        results_df = self.bm25.transform(query_df)
        
        # Step 2: Add text
        results_df = self._add_text(results_df)
        
        # Step 3: TCT-ColBERT reranking
        results_df = self.tct_reranker.transform(results_df)
        
        return results_df
    
    def retrieve(
        self,
        query: str,
        qid: str,
        top_k: int = 100,
        **kwargs
    ) -> RetrieverResult:
        """Retrieve using manual pipeline: BM25 >> text >> TCT-ColBERT."""
        import pandas as pd
        import re
        
        start = time.time()
        
        # Sanitize query
        clean_query = re.sub(r"[^a-zA-Z0-9\s]", " ", query)
        clean_query = re.sub(r"\s+", " ", clean_query).strip()
        
        query_df = pd.DataFrame([{"qid": qid, "query": clean_query}])
        
        # Run pipeline
        results_df = self._run_pipeline(query_df)
        results_df = results_df.head(top_k)
        
        results = []
        for rank, (_, row) in enumerate(results_df.iterrows(), start=1):
            results.append((
                str(row["docno"]),
                float(row["score"]),
                rank
            ))
        
        latency = (time.time() - start) * 1000
        
        return RetrieverResult(
            qid=qid,
            results=results,
            retriever_name=self.name,
            latency_ms=latency,
            metadata={"model": self.TCT_MODEL, "first_stage_k": self.first_stage_k}
        )
    
    def retrieve_batch(
        self,
        queries: Dict[str, str],
        top_k: int = 100,
        **kwargs
    ) -> Dict[str, RetrieverResult]:
        """Batch retrieval using manual pipeline: BM25 >> text >> TCT-ColBERT."""
        import pandas as pd
        import re
        
        start = time.time()
        n_queries = len(queries)
        
        print(f"[BM25_TCT] Batch retrieval: {n_queries} queries")
        
        # Build query DataFrame
        query_df = pd.DataFrame([
            {
                "qid": qid, 
                "query": re.sub(r"\s+", " ", re.sub(r"[^a-zA-Z0-9\s]", " ", text)).strip()
            }
            for qid, text in queries.items()
        ])
        
        # Run pipeline
        results_df = self._run_pipeline(query_df)
        
        # Group by query and build results (OPTIMIZED: O(n) instead of O(n²))
        results = {}
        grouped = results_df.groupby("qid")
        for qid, group in grouped:
            qid_results = group.head(top_k)
            
            doc_list = []
            for rank, (_, row) in enumerate(qid_results.iterrows(), start=1):
                doc_list.append((
                    str(row["docno"]),
                    float(row["score"]),
                    rank
                ))
            
            results[qid] = RetrieverResult(
                qid=qid,
                results=doc_list,
                retriever_name=self.name,
                latency_ms=0,
                metadata={"model": self.TCT_MODEL}
            )
        
        total_time = (time.time() - start) * 1000
        print(f"[BM25_TCT] Batch complete: {n_queries} queries in {total_time:.1f}ms")
        
        return results

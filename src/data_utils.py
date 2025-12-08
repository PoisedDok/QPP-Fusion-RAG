#!/usr/bin/env python3
"""
Incoming: BEIR dataset files --- {corpus.jsonl, queries.jsonl, qrels/test.tsv, TREC .res}
Processing: data loading --- {parsing, indexing}
Outgoing: Python data structures --- {Dict, List}

Centralized Data Loading Utilities
----------------------------------
Single source of truth for loading BEIR datasets, run files, and qrels.
All scripts import from here to avoid duplication.

STRICT: No redundant implementations. One loader per data type.
"""

import os
import json
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple, Optional, Union


# =============================================================================
# Corpus Loading
# =============================================================================

class LazyCorpus:
    """
    Lazy-loading corpus that only loads documents when accessed.
    Memory-efficient: builds offset index, loads docs on-demand.
    
    Usage:
        corpus = LazyCorpus("/path/to/beir/dataset")
        doc = corpus["doc123"]  # Loads from disk
        doc = corpus.get("doc123", default=None)
    """
    
    def __init__(self, corpus_path: str):
        """
        Args:
            corpus_path: Path to BEIR dataset directory containing corpus.jsonl
        """
        self.corpus_file = os.path.join(corpus_path, "corpus.jsonl")
        self._cache: Dict[str, Dict[str, str]] = {}
        self._offsets: Dict[str, int] = {}
        self._build_offset_index()
    
    def _build_offset_index(self):
        """Build document ID to file offset map."""
        with open(self.corpus_file, 'r', encoding='utf-8') as f:
            offset = 0
            for line in f:
                doc = json.loads(line)
                doc_id = doc.get("_id", "")
                self._offsets[doc_id] = offset
                offset += len(line.encode('utf-8'))
    
    def get(self, doc_id: str, default=None) -> Optional[Dict[str, str]]:
        """Get document by ID (loads on-demand)."""
        if doc_id in self._cache:
            return self._cache[doc_id]
        
        if doc_id not in self._offsets:
            return default
        
        with open(self.corpus_file, 'r', encoding='utf-8') as f:
            f.seek(self._offsets[doc_id])
            doc = json.loads(f.readline())
            result = {
                "text": doc.get("text", ""),
                "title": doc.get("title", "")
            }
            self._cache[doc_id] = result
            return result
    
    def __getitem__(self, doc_id: str) -> Dict[str, str]:
        result = self.get(doc_id)
        if result is None:
            raise KeyError(doc_id)
        return result
    
    def __contains__(self, doc_id: str) -> bool:
        return doc_id in self._offsets
    
    def __len__(self) -> int:
        return len(self._offsets)


def load_corpus(
    corpus_path: str, 
    lazy: bool = True, 
    limit: int = None
) -> Union[LazyCorpus, Dict[str, Dict[str, str]]]:
    """
    Load BEIR corpus.
    
    Args:
        corpus_path: Path to BEIR dataset directory
        lazy: If True, return LazyCorpus (memory-efficient). 
              If False, load all into memory.
        limit: Limit number of documents (only for eager loading)
    
    Returns:
        LazyCorpus or Dict[doc_id, {"text": ..., "title": ...}]
    """
    if lazy:
        return LazyCorpus(corpus_path)
    
    corpus = {}
    corpus_file = os.path.join(corpus_path, "corpus.jsonl")
    
    with open(corpus_file, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if limit and i >= limit:
                break
            doc = json.loads(line)
            doc_id = doc.get("_id", str(i))
            corpus[doc_id] = {
                "text": doc.get("text", ""),
                "title": doc.get("title", "")
            }
    
    return corpus


# =============================================================================
# Query Loading
# =============================================================================

def load_queries(
    corpus_path: str, 
    split: str = "test"
) -> Dict[str, str]:
    """
    Load BEIR queries, filtered by split.
    
    Args:
        corpus_path: Path to BEIR dataset directory
        split: Split to load (test/dev/train). Filters by qrels file.
    
    Returns:
        Dict[query_id, query_text]
    """
    queries = {}
    queries_file = os.path.join(corpus_path, "queries.jsonl")
    qrels_file = os.path.join(corpus_path, "qrels", f"{split}.tsv")
    
    # Get query IDs from qrels to filter
    split_qids = set()
    if os.path.exists(qrels_file):
        with open(qrels_file, 'r') as f:
            next(f, None)  # Skip header
            for line in f:
                parts = line.strip().split('\t')
                if parts:
                    split_qids.add(parts[0])
    
    # Load queries (filtered if qrels exist)
    with open(queries_file, 'r', encoding='utf-8') as f:
        for line in f:
            q = json.loads(line)
            qid = q.get("_id", "")
            if not split_qids or qid in split_qids:
                queries[qid] = q.get("text", "")
    
    return queries


# =============================================================================
# Qrels Loading
# =============================================================================

def load_qrels(
    qrels_path: Union[str, Path]
) -> Dict[str, Dict[str, int]]:
    """
    Load qrels from TSV file.
    
    Args:
        qrels_path: Path to qrels TSV file (e.g., qrels/test.tsv)
    
    Returns:
        Dict[query_id, Dict[doc_id, relevance]]
    """
    qrels = defaultdict(dict)
    
    with open(qrels_path, 'r') as f:
        header = next(f, None)  # Skip header
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) >= 3:
                qid, docid, rel = parts[0], parts[1], int(parts[2])
                qrels[qid][docid] = rel
    
    return dict(qrels)


# =============================================================================
# Run File Loading
# =============================================================================

def load_run_file(
    run_path: Union[str, Path]
) -> Dict[str, List[Tuple[str, float, int]]]:
    """
    Load TREC-format run file.
    
    Format: qid Q0 docid rank score tag
    
    Args:
        run_path: Path to .res file
    
    Returns:
        Dict[query_id, List[(doc_id, score, rank)]] sorted by rank
    """
    runs = defaultdict(list)
    
    with open(run_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 6:
                qid, _, docid, rank, score, tag = parts[:6]
                runs[qid].append((docid, float(score), int(rank)))
    
    # Sort by rank
    for qid in runs:
        runs[qid].sort(key=lambda x: x[2])
    
    return dict(runs)


def load_run_as_dict(
    run_path: Union[str, Path]
) -> Dict[str, Dict[str, float]]:
    """
    Load TREC run file as nested dict.
    
    Args:
        run_path: Path to .res file
    
    Returns:
        Dict[query_id, Dict[doc_id, score]]
    """
    runs = defaultdict(dict)
    
    with open(run_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 5:
                qid, _, docid, rank, score = parts[:5]
                runs[qid][docid] = float(score)
    
    return dict(runs)


# =============================================================================
# QPP Score Loading
# =============================================================================

def load_qpp_scores(
    qpp_path: Union[str, Path]
) -> Dict[str, Dict[str, List[float]]]:
    """
    Load QPP scores from directory.
    
    Args:
        qpp_path: Directory containing .mmnorm.qpp files
    
    Returns:
        Dict[query_id, Dict[retriever_name, List[13 QPP scores]]]
    """
    qpp_data = defaultdict(dict)
    qpp_path = Path(qpp_path)
    
    files = list(qpp_path.glob("*.mmnorm.qpp"))
    if not files:
        raise FileNotFoundError(f"No .mmnorm.qpp files in {qpp_path}")
    
    for qpp_file in files:
        retriever = qpp_file.stem.replace(".res.mmnorm", "")
        
        with open(qpp_file, 'r') as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) < 2:
                    continue
                qid = parts[0]
                scores = [float(x) for x in parts[1:]]
                qpp_data[qid][retriever] = scores
    
    return dict(qpp_data)


# =============================================================================
# Utility Functions
# =============================================================================

def get_model_safe_name(model: str) -> str:
    """Convert model name to filesystem-safe name."""
    return model.replace("/", "_").replace(":", "_")


def detect_dataset(path: str) -> str:
    """Detect dataset name from path."""
    path_lower = path.lower()
    if "hotpot" in path_lower:
        return "hotpotqa"
    elif "nq" in path_lower or "natural" in path_lower:
        return "nq"
    elif "msmarco" in path_lower:
        return "msmarco"
    else:
        return "unknown"

#!/usr/bin/env python3
"""
Incoming: index, corpus, queries --- {pt.Index, Dict, Dict}
Processing: retrieval --- {5 jobs: BM25, TCT, Splade, BM25_TCT, BM25_MonoT5}
Outgoing: TREC run files --- {.res files}

Step 2: Run Retrievers
----------------------
Runs retrievers sequentially with memory management.
Optimized for Mac M4 16GB:
- Loads corpus only when needed
- Clears memory between retrievers
- Uses disk caching for embeddings

Usage:
    python scripts/02_retrieve.py --corpus_path /data/beir/datasets/nq
    python scripts/02_retrieve.py --corpus_path /data/beir/datasets/nq --retrievers BM25
    python scripts/02_retrieve.py --corpus_path /data/beir/datasets/nq --limit 10000
"""

import os
import sys
import gc
import json
import argparse
import time
from pathlib import Path
from typing import Dict, Optional

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Set cache paths to Disk-D before importing ML libs
from src.cache_config import CACHE_ROOT


def load_corpus(corpus_path: str, limit: int = None) -> Dict[str, Dict[str, str]]:
    """Load BEIR corpus."""
    corpus = {}
    corpus_file = os.path.join(corpus_path, "corpus.jsonl")
    
    print(f"[02_retrieve] Loading corpus...")
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
            if (i + 1) % 500000 == 0:
                print(f"  Loaded {i + 1} documents...")
    
    print(f"[02_retrieve] Corpus: {len(corpus)} documents")
    return corpus


def load_queries(corpus_path: str, split: str = "test") -> Dict[str, str]:
    """Load BEIR queries, filtered by split (test/dev/train)."""
    queries = {}
    queries_file = os.path.join(corpus_path, "queries.jsonl")
    qrels_file = os.path.join(corpus_path, "qrels", f"{split}.tsv")
    
    # Get test query IDs from qrels
    test_qids = set()
    if os.path.exists(qrels_file):
        with open(qrels_file, 'r') as f:
            next(f)  # Skip header
            for line in f:
                parts = line.strip().split('\t')
                if parts:
                    test_qids.add(parts[0])
        print(f"[02_retrieve] Found {len(test_qids)} unique {split} query IDs in qrels")
    else:
        print(f"[02_retrieve] No qrels/{split}.tsv found, loading all queries")
    
    # Load queries (filtered by test QIDs if available)
    with open(queries_file, 'r', encoding='utf-8') as f:
        for line in f:
            q = json.loads(line)
            qid = q.get("_id", "")
            if not test_qids or qid in test_qids:
                queries[qid] = q.get("text", "")
    
    print(f"[02_retrieve] Loaded {len(queries)} {split} queries")
    return queries


def write_run(results: Dict, output_path: str, retriever_name: str, normalize: bool = False):
    """Write results to TREC run file."""
    from src.retrievers.base import BaseRetriever
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, 'w') as f:
        for qid in sorted(results.keys(), key=lambda x: int(x.replace("test", "")) if x.startswith("test") else x):
            result = results[qid]
            docs = result.results
            
            if normalize:
                docs = BaseRetriever.normalize_scores(docs)
            
            for docno, score, rank in docs:
                f.write(f"{qid} Q0 {docno} {rank} {score:.6f} {retriever_name}\n")
    
    print(f"[02_retrieve] Wrote {output_path}")


def clear_memory():
    """Force garbage collection and clear caches."""
    gc.collect()
    try:
        import torch
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
        elif torch.cuda.is_available():
            torch.cuda.empty_cache()
    except:
        pass


def run_bm25(index_path: str, queries: Dict[str, str], runs_dir: Path, top_k: int):
    """Run BM25 retriever."""
    from src.retrievers import BM25Retriever
    
    print(f"\n[02_retrieve] === BM25 ===")
    start = time.time()
    
    retriever = BM25Retriever(index_path)
    results = retriever.retrieve_batch(queries, top_k=top_k)
    
    write_run(results, str(runs_dir / "BM25.res"), "BM25", normalize=False)
    write_run(results, str(runs_dir / "BM25.norm.res"), "BM25", normalize=True)
    
    print(f"[02_retrieve] BM25 completed in {time.time() - start:.1f}s")
    
    del retriever, results
    clear_memory()


def run_tct_colbert(corpus: Dict, queries: Dict[str, str], runs_dir: Path, cache_dir: Path, top_k: int):
    """Run TCT-ColBERT retriever with checkpointing."""
    from src.retrievers import TCTColBERTRetriever
    
    print(f"\n[02_retrieve] === TCT-ColBERT ===")
    start = time.time()
    
    checkpoint_path = runs_dir / "TCT-ColBERT.checkpoint.jsonl"
    
    retriever = TCTColBERTRetriever(
        corpus,
        cache_dir=str(cache_dir),
        batch_size=64,
        use_fp16=True
    )
    results = retriever.retrieve_batch(
        queries, 
        top_k=top_k,
        checkpoint_path=str(checkpoint_path),
        mini_batch_size=100
    )
    
    write_run(results, str(runs_dir / "TCT-ColBERT.res"), "TCT-ColBERT", normalize=False)
    write_run(results, str(runs_dir / "TCT-ColBERT.norm.res"), "TCT-ColBERT", normalize=True)
    
    if checkpoint_path.exists():
        checkpoint_path.unlink()
        print(f"[02_retrieve] Removed checkpoint file")
    
    print(f"[02_retrieve] TCT-ColBERT completed in {time.time() - start:.1f}s")
    
    del retriever, results
    clear_memory()


def run_splade(queries: Dict[str, str], runs_dir: Path, top_k: int, dataset: str = "nq"):
    """Run SPLADE retriever with checkpointing."""
    from src.retrievers import SpladeRetriever
    
    print(f"\n[02_retrieve] === SPLADE (Pyserini Pre-built) ===")
    start = time.time()
    
    checkpoint_path = runs_dir / "Splade.checkpoint.jsonl"
    index_name = f"beir-v1.0.0-{dataset}.splade-pp-ed"
    
    retriever = SpladeRetriever(index_name=index_name, threads=10)
    results = retriever.retrieve_batch(
        queries, 
        top_k=top_k,
        checkpoint_path=str(checkpoint_path),
        mini_batch_size=500  # SPLADE is fast
    )
    
    write_run(results, str(runs_dir / "Splade.res"), "Splade", normalize=False)
    write_run(results, str(runs_dir / "Splade.norm.res"), "Splade", normalize=True)
    
    if checkpoint_path.exists():
        checkpoint_path.unlink()
        print(f"[02_retrieve] Removed checkpoint file")
    
    print(f"[02_retrieve] SPLADE completed in {time.time() - start:.1f}s")
    
    del retriever, results
    clear_memory()


def run_bge(queries: Dict[str, str], runs_dir: Path, top_k: int, dataset: str = "nq"):
    """Run BGE retriever with checkpointing."""
    from src.retrievers import BGERetriever
    
    print(f"\n[02_retrieve] === BGE (Pyserini Pre-built FAISS) ===")
    start = time.time()
    
    checkpoint_path = runs_dir / "BGE.checkpoint.jsonl"
    index_name = f"beir-v1.0.0-{dataset}.bge-base-en-v1.5"
    
    retriever = BGERetriever(index_name=index_name, threads=10, use_mps=True)
    results = retriever.retrieve_batch(
        queries, 
        top_k=top_k,
        checkpoint_path=str(checkpoint_path),
        mini_batch_size=200  # BGE is fast
    )
    
    write_run(results, str(runs_dir / "BGE.res"), "BGE", normalize=False)
    write_run(results, str(runs_dir / "BGE.norm.res"), "BGE", normalize=True)
    
    if checkpoint_path.exists():
        checkpoint_path.unlink()
        print(f"[02_retrieve] Removed checkpoint file")
    
    print(f"[02_retrieve] BGE completed in {time.time() - start:.1f}s")
    
    del retriever, results
    clear_memory()


def run_bm25_tct(index_path: str, corpus_path: str, queries: Dict[str, str], runs_dir: Path, top_k: int):
    """Run BM25>>TCT-ColBERT retriever with lazy corpus loading and checkpointing."""
    from src.retrievers import BM25TCTRetriever
    
    print(f"\n[02_retrieve] === BM25>>TCT-ColBERT ===")
    start = time.time()
    
    # Checkpoint path for crash recovery
    checkpoint_path = runs_dir / "BM25_TCT.checkpoint.jsonl"
    
    # Lazy loading: pass corpus_path, not corpus dict
    # first_stage_k=100 (not 500) for speed
    retriever = BM25TCTRetriever(
        index_path, 
        corpus_path, 
        first_stage_k=100,  # Reduced from 500 for 5x speedup
        tct_batch_size=64   # TCT encoding batch size
    )
    results = retriever.retrieve_batch(
        queries, 
        top_k=top_k,
        checkpoint_path=str(checkpoint_path),
        mini_batch_size=5  # Small batches for MPS efficiency
    )
    
    write_run(results, str(runs_dir / "BM25_TCT.res"), "BM25_TCT", normalize=False)
    write_run(results, str(runs_dir / "BM25_TCT.norm.res"), "BM25_TCT", normalize=True)
    
    # Clean up checkpoint after successful completion
    if checkpoint_path.exists():
        checkpoint_path.unlink()
        print(f"[02_retrieve] Removed checkpoint file")
    
    print(f"[02_retrieve] BM25>>TCT completed in {time.time() - start:.1f}s")
    
    del retriever, results
    clear_memory()


def run_bm25_monot5(index_path: str, corpus_path: str, queries: Dict[str, str], runs_dir: Path, top_k: int):
    """Run BM25>>MonoT5 retriever with checkpointing."""
    from src.retrievers import BM25MonoT5Retriever
    
    print(f"\n[02_retrieve] === BM25>>MonoT5 ===")
    start = time.time()
    
    checkpoint_path = runs_dir / "BM25_MonoT5.checkpoint.jsonl"
    
    retriever = BM25MonoT5Retriever(
        index_path, 
        corpus_path, 
        first_stage_k=100,
        ce_batch_size=256  # CrossEncoder batch size
    )
    results = retriever.retrieve_batch(
        queries, 
        top_k=top_k,
        checkpoint_path=str(checkpoint_path),
        mini_batch_size=10  # CrossEncoder is slower, smaller batches
    )
    
    write_run(results, str(runs_dir / "BM25_MonoT5.res"), "BM25_MonoT5", normalize=False)
    write_run(results, str(runs_dir / "BM25_MonoT5.norm.res"), "BM25_MonoT5", normalize=True)
    
    if checkpoint_path.exists():
        checkpoint_path.unlink()
        print(f"[02_retrieve] Removed checkpoint file")
    
    print(f"[02_retrieve] BM25>>MonoT5 completed in {time.time() - start:.1f}s")
    
    del retriever, results
    clear_memory()


def main():
    parser = argparse.ArgumentParser(description="Step 2: Run Retrievers")
    parser.add_argument("--corpus_path", required=True, help="Path to BEIR dataset")
    parser.add_argument("--index_path", default=None, help="PyTerrier index path")
    parser.add_argument("--output_dir", default=None, help="Output directory")
    parser.add_argument("--retrievers", default="BM25,TCT-ColBERT,BM25_TCT,BM25_MonoT5",
                        help="Comma-separated retriever names (Splade excluded by default)")
    parser.add_argument("--top_k", type=int, default=100, help="Number of docs to retrieve")
    parser.add_argument("--limit", type=int, default=None, help="Limit corpus size")
    parser.add_argument("--splade_max_docs", type=int, default=None, 
                        help="Max docs for SPLADE (memory safety)")
    args = parser.parse_args()
    
    # Setup paths
    output_dir = Path(args.output_dir) if args.output_dir else PROJECT_ROOT / "data" / "nq"
    runs_dir = output_dir / "runs"
    cache_dir = output_dir / "cache"
    # Use absolute path for PyTerrier index to avoid CWD issues
    if args.index_path:
        index_path = str(Path(args.index_path).resolve())
    else:
        index_path = str((output_dir / "index" / "pyterrier").resolve())
    
    os.makedirs(runs_dir, exist_ok=True)
    os.makedirs(cache_dir, exist_ok=True)
    
    # Parse retrievers
    retrievers = [r.strip() for r in args.retrievers.split(",")]
    print(f"[02_retrieve] Retrievers: {retrievers}")
    print(f"[02_retrieve] Cache dir: {cache_dir}")
    
    # Load queries first (small)
    queries = load_queries(args.corpus_path)
    
    # Check which retrievers need corpus in memory
    # Note: BM25_TCT, BM25_MonoT5, and Splade use lazy/pre-built loading
    needs_corpus = any(r in retrievers for r in ["TCT-ColBERT"])
    
    corpus = None
    if needs_corpus:
        corpus = load_corpus(args.corpus_path, args.limit)
    
    # Run each retriever
    try:
        if "BM25" in retrievers:
            run_bm25(index_path, queries, runs_dir, args.top_k)
        
        if "TCT-ColBERT" in retrievers:
            run_tct_colbert(corpus, queries, runs_dir, cache_dir, args.top_k)
        
        if "Splade" in retrievers:
            # Detect dataset from corpus_path
            dataset = "hotpotqa" if "hotpot" in args.corpus_path.lower() else "nq"
            run_splade(queries, runs_dir, args.top_k, dataset=dataset)
        
        if "BGE" in retrievers:
            # Detect dataset from corpus_path
            dataset = "hotpotqa" if "hotpot" in args.corpus_path.lower() else "nq"
            run_bge(queries, runs_dir, args.top_k, dataset=dataset)
        
        if "BM25_TCT" in retrievers:
            # Uses lazy corpus loading - pass path not dict
            run_bm25_tct(index_path, args.corpus_path, queries, runs_dir, args.top_k)
        
        if "BM25_MonoT5" in retrievers:
            # Uses lazy corpus loading - pass path not dict
            run_bm25_monot5(index_path, args.corpus_path, queries, runs_dir, args.top_k)
            
    except Exception as e:
        print(f"[02_retrieve] ERROR: {e}")
        import traceback
        traceback.print_exc()
        raise
    
    print(f"\n=== Step 2 Complete ===")
    print(f"Runs directory: {runs_dir}")
    for f in sorted(runs_dir.glob("*.res")):
        print(f"  {f.name}")


if __name__ == "__main__":
    main()

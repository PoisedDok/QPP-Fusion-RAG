#!/usr/bin/env python3
"""
Incoming: BEIR corpus (corpus.jsonl) --- {JSON lines, Pyserini pre-built}
Processing: indexing --- {2 jobs: PyTerrier build, Pyserini download}
Outgoing: indexes --- {PyTerrier index, Pyserini cache}

Step 1: Index NQ Corpus
-----------------------
Creates/downloads indexes for retrieval:
- PyTerrier: Builds BM25 inverted index from BEIR corpus (streaming, memory-efficient)
- SPLADE: Downloads Pyserini pre-built index (~2GB, no encoding needed)

Usage:
    python scripts/01_index.py --corpus_path /data/beir/datasets/nq
    python scripts/01_index.py --corpus_path /data/beir/datasets/nq --indexes pyterrier
    python scripts/01_index.py --corpus_path /data/beir/datasets/nq --indexes splade
    python scripts/01_index.py --corpus_path /data/beir/datasets/nq --indexes pyterrier,splade
"""

import os
import sys
import json
import argparse
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Set cache paths to Disk-D before importing ML libs
from src.cache_config import CACHE_ROOT
print(f"[01_index] Cache root: {CACHE_ROOT}")


# Pre-built index registry (use dot notation: beir-v1.0.0-nq.splade-pp-ed)
PREBUILT_INDEXES = {
    "splade": {
        "index_name": "beir-v1.0.0-nq.splade-pp-ed",
        "encoder": "naver/splade-cocondenser-ensembledistil",
        "searcher_class": "LuceneImpactSearcher",
        "description": "SPLADE++ EnsembleDistil for BEIR-NQ (~2GB download)"
    },
    "splade-v3": {
        "index_name": "beir-v1.0.0-nq.splade-v3",
        "encoder": "naver/splade-v3",
        "searcher_class": "LuceneImpactSearcher",
        "description": "SPLADE v3 for BEIR-NQ"
    },
    "bm25-pyserini": {
        "index_name": "beir-v1.0.0-nq.flat",
        "encoder": None,
        "searcher_class": "LuceneSearcher",
        "description": "Pyserini BM25 for BEIR-NQ"
    },
    "contriever": {
        "index_name": "beir-v1.0.0-nq.contriever-msmarco",
        "encoder": "facebook/contriever-msmarco",
        "searcher_class": "FaissSearcher",
        "description": "Contriever-MSMARCO dense for BEIR-NQ"
    },
    "bge": {
        "index_name": "beir-v1.0.0-nq.bge-base-en-v1.5",
        "encoder": "BAAI/bge-base-en-v1.5",
        "searcher_class": "FaissSearcher",
        "description": "BGE-base dense for BEIR-NQ"
    }
}


def stream_corpus(corpus_path: str, limit: int = None):
    """Stream BEIR corpus without loading all into memory."""
    corpus_file = os.path.join(corpus_path, "corpus.jsonl")
    
    print(f"[01_index] Streaming corpus from {corpus_file}")
    count = 0
    
    with open(corpus_file, 'r', encoding='utf-8') as f:
        for line in f:
            if limit and count >= limit:
                break
            
            doc = json.loads(line)
            doc_id = doc.get("_id", str(count))
            text = (doc.get("title", "") + " " + doc.get("text", "")).strip()
            
            yield {"docno": doc_id, "text": text}
            
            count += 1
            if count % 500000 == 0:
                print(f"  Processed {count} documents...")
    
    print(f"[01_index] Total: {count} documents")


def count_docs(corpus_path: str) -> int:
    """Count documents in corpus (for progress tracking)."""
    corpus_file = os.path.join(corpus_path, "corpus.jsonl")
    count = 0
    with open(corpus_file, 'r') as f:
        for _ in f:
            count += 1
    return count


def build_pyterrier_index(corpus_path: str, index_path: str, limit: int = None) -> str:
    """Build PyTerrier inverted index with streaming."""
    import pyterrier as pt
    if not pt.started():
        pt.init()
    
    print(f"[01_index] Building PyTerrier index at {index_path}")
    os.makedirs(index_path, exist_ok=True)
    
    start = time.time()
    
    # Use streaming iterator
    indexer = pt.IterDictIndexer(
        index_path,
        meta={"docno": 100},
        verbose=True
    )
    
    index_ref = indexer.index(stream_corpus(corpus_path, limit))
    elapsed = time.time() - start
    
    print(f"[01_index] PyTerrier index built in {elapsed:.1f}s")
    print(f"[01_index] Index ref: {index_ref}")
    
    # Verify index
    index = pt.IndexFactory.of(index_ref)
    stats = index.getCollectionStatistics()
    print(f"[01_index] Index stats: {stats.getNumberOfDocuments()} docs, "
          f"{stats.getNumberOfTokens()} tokens")
    
    return str(index_ref)


def download_prebuilt_index(index_key: str) -> bool:
    """Download and cache Pyserini pre-built index."""
    if index_key not in PREBUILT_INDEXES:
        print(f"[01_index] ERROR: Unknown pre-built index: {index_key}")
        print(f"[01_index] Available: {list(PREBUILT_INDEXES.keys())}")
        return False
    
    config = PREBUILT_INDEXES[index_key]
    index_name = config["index_name"]
    encoder = config["encoder"]
    searcher_class = config["searcher_class"]
    
    print(f"\n[01_index] === Downloading Pre-built Index: {index_key} ===")
    print(f"[01_index] {config['description']}")
    print(f"[01_index] Index: {index_name}")
    if encoder:
        print(f"[01_index] Encoder: {encoder}")
    
    start = time.time()
    
    try:
        if searcher_class == "LuceneImpactSearcher":
            from pyserini.search.lucene import LuceneImpactSearcher
            print(f"[01_index] Initializing LuceneImpactSearcher (downloads on first use)...")
            searcher = LuceneImpactSearcher.from_prebuilt_index(index_name, encoder)
            
        elif searcher_class == "LuceneSearcher":
            from pyserini.search.lucene import LuceneSearcher
            print(f"[01_index] Initializing LuceneSearcher (downloads on first use)...")
            searcher = LuceneSearcher.from_prebuilt_index(index_name)
            
        elif searcher_class == "FaissSearcher":
            from pyserini.search.faiss import FaissSearcher
            print(f"[01_index] Initializing FaissSearcher (downloads on first use)...")
            searcher = FaissSearcher.from_prebuilt_index(index_name, encoder)
        
        else:
            print(f"[01_index] ERROR: Unknown searcher class: {searcher_class}")
            return False
        
        # Test search to verify
        print(f"[01_index] Verifying index with test query...")
        hits = searcher.search("test query", k=5)
        
        elapsed = time.time() - start
        print(f"[01_index] SUCCESS: {index_key} index ready ({elapsed:.1f}s)")
        print(f"[01_index] Test search returned {len(hits)} hits")
        
        # Cleanup
        del searcher
        
        return True
        
    except Exception as e:
        print(f"[01_index] ERROR downloading {index_key}: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    parser = argparse.ArgumentParser(description="Step 1: Index NQ Corpus")
    parser.add_argument("--corpus_path", required=True, help="Path to BEIR dataset")
    parser.add_argument("--index_path", default=None, help="Output index path (for PyTerrier)")
    parser.add_argument("--limit", type=int, default=None, help="Limit corpus size (PyTerrier only)")
    parser.add_argument("--indexes", default="pyterrier",
                        help="Comma-separated indexes to build/download: pyterrier,splade,contriever,bge")
    parser.add_argument("--list-prebuilt", action="store_true", help="List available pre-built indexes")
    parser.add_argument("--force", action="store_true", help="Force rebuild even if exists")
    args = parser.parse_args()
    
    # List pre-built indexes
    if args.list_prebuilt:
        print("\n=== Available Pre-built Indexes ===\n")
        for key, config in PREBUILT_INDEXES.items():
            print(f"  {key}:")
            print(f"    Index: {config['index_name']}")
            print(f"    {config['description']}")
            print()
        return
    
    # Parse requested indexes
    requested = [idx.strip().lower() for idx in args.indexes.split(",")]
    print(f"[01_index] Requested indexes: {requested}")
    
    # Default paths
    output_dir = PROJECT_ROOT / "data" / "nq"
    index_path = args.index_path or str(output_dir / "index" / "pyterrier")
    
    results = {}
    
    # Build PyTerrier index if requested
    if "pyterrier" in requested:
        print(f"\n[01_index] === PyTerrier BM25 Index ===")
        
        if os.path.exists(index_path) and os.listdir(index_path) and not args.force:
            print(f"[01_index] PyTerrier index already exists at {index_path}")
            print("[01_index] Use --force to rebuild")
            results["pyterrier"] = "EXISTS"
        else:
            build_pyterrier_index(args.corpus_path, index_path, args.limit)
            results["pyterrier"] = "BUILT"
    
    # Download pre-built indexes
    for idx in requested:
        if idx == "pyterrier":
            continue  # Already handled
        
        if idx in PREBUILT_INDEXES:
            success = download_prebuilt_index(idx)
            results[idx] = "READY" if success else "FAILED"
        else:
            print(f"[01_index] WARNING: Unknown index type: {idx}")
            print(f"[01_index] Use --list-prebuilt to see available options")
            results[idx] = "UNKNOWN"
    
    # Summary
    print(f"\n=== Step 1 Complete ===")
    print(f"Results:")
    for idx, status in results.items():
        print(f"  {idx}: {status}")
    
    if "pyterrier" in results:
        print(f"\nPyTerrier index: {index_path}")
    
    if any(k in results for k in PREBUILT_INDEXES):
        print(f"\nPre-built indexes cached in: ~/.cache/pyserini/")


if __name__ == "__main__":
    main()

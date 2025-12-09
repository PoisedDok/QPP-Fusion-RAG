#!/usr/bin/env python3
"""
Incoming: BEIR corpus or Pyserini pre-built --- {corpus.jsonl or remote index}
Processing: indexing --- {3 jobs: PyTerrier build, Pyserini download, HNSW build}
Outgoing: search indexes --- {PyTerrier index, FAISS flat, HNSW segments}

Step 1: Build/Download Search Indexes
-------------------------------------
Creates indexes for retrieval:
- PyTerrier: BM25 inverted index from BEIR corpus
- Pyserini: Downloads pre-built indexes (SPLADE, BGE, Contriever)
- HNSW: Builds segmented HNSW from BGE FAISS for fast dense search

Usage:
    # Download BGE FAISS + build HNSW (recommended for dense retrieval)
    python scripts/01_index.py --dataset hotpotqa --indexes bge --hnsw
    
    # PyTerrier BM25 only
    python scripts/01_index.py --dataset nq --corpus_path /data/beir/nq --indexes pyterrier
    
    # Multiple indexes
    python scripts/01_index.py --dataset nq --indexes bge,splade
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.cache_config import CACHE_ROOT

# Pre-built index registry
PREBUILT_INDEXES = {
    "splade": {
        "index_name": "beir-v1.0.0-{dataset}.splade-pp-ed",
        "encoder": "naver/splade-cocondenser-ensembledistil",
        "searcher_class": "LuceneImpactSearcher",
        "description": "SPLADE++ EnsembleDistil"
    },
    "bge": {
        "index_name": "beir-v1.0.0-{dataset}.bge-base-en-v1.5",
        "encoder": "BAAI/bge-base-en-v1.5",
        "searcher_class": "FaissSearcher",
        "description": "BGE-base dense vectors"
    },
    "contriever": {
        "index_name": "beir-v1.0.0-{dataset}.contriever-msmarco",
        "encoder": "facebook/contriever-msmarco",
        "searcher_class": "FaissSearcher",
        "description": "Contriever-MSMARCO dense"
    },
    "bm25-pyserini": {
        "index_name": "beir-v1.0.0-{dataset}.flat",
        "encoder": None,
        "searcher_class": "LuceneSearcher",
        "description": "Pyserini BM25"
    }
}


def stream_corpus(corpus_path: str, limit: int = None):
    """Stream BEIR corpus without loading all into memory."""
    corpus_file = os.path.join(corpus_path, "corpus.jsonl")
    print(f"[Index] Streaming from {corpus_file}")
    
    count = 0
    with open(corpus_file, 'r', encoding='utf-8') as f:
        for line in f:
            if limit and count >= limit:
                break
            doc = json.loads(line)
            yield {
                "docno": doc.get("_id", str(count)),
                "text": (doc.get("title", "") + " " + doc.get("text", "")).strip()
            }
            count += 1
            if count % 500000 == 0:
                print(f"  Processed {count:,} documents...")
    
    print(f"[Index] Total: {count:,} documents")


def build_pyterrier_index(corpus_path: str, index_path: str, limit: int = None) -> str:
    """Build PyTerrier BM25 inverted index."""
    import pyterrier as pt
    if not pt.started():
        pt.init()
    
    print(f"[Index] Building PyTerrier index at {index_path}")
    os.makedirs(index_path, exist_ok=True)
    
    t0 = time.time()
    indexer = pt.IterDictIndexer(index_path, meta={"docno": 100}, verbose=True)
    index_ref = indexer.index(stream_corpus(corpus_path, limit))
    
    index = pt.IndexFactory.of(index_ref)
    stats = index.getCollectionStatistics()
    print(f"[Index] Built in {time.time()-t0:.1f}s: "
          f"{stats.getNumberOfDocuments():,} docs, {stats.getNumberOfTokens():,} tokens")
    
    return str(index_ref)


def download_prebuilt_index(index_key: str, dataset: str) -> bool:
    """Download Pyserini pre-built index."""
    if index_key not in PREBUILT_INDEXES:
        print(f"[Index] Unknown index: {index_key}")
        return False
    
    config = PREBUILT_INDEXES[index_key]
    index_name = config["index_name"].format(dataset=dataset)
    encoder = config["encoder"]
    searcher_class = config["searcher_class"]
    
    print(f"\n[Index] Downloading {index_key}: {config['description']}")
    print(f"  Index: {index_name}")
    
    t0 = time.time()
    try:
        if searcher_class == "LuceneImpactSearcher":
            from pyserini.search.lucene import LuceneImpactSearcher
            searcher = LuceneImpactSearcher.from_prebuilt_index(index_name, encoder)
        elif searcher_class == "LuceneSearcher":
            from pyserini.search.lucene import LuceneSearcher
            searcher = LuceneSearcher.from_prebuilt_index(index_name)
        elif searcher_class == "FaissSearcher":
            from pyserini.search.faiss import FaissSearcher
            searcher = FaissSearcher.from_prebuilt_index(index_name, encoder)
        else:
            print(f"[Index] Unknown searcher: {searcher_class}")
            return False
        
        # Verify
        hits = searcher.search("test query", k=5)
        del searcher
        
        print(f"[Index] Ready ({time.time()-t0:.1f}s, {len(hits)} test hits)")
        return True
        
    except Exception as e:
        print(f"[Index] ERROR: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Step 1: Build/Download Search Indexes",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # BGE dense with HNSW (fast retrieval)
  python scripts/01_index.py --dataset hotpotqa --indexes bge --hnsw
  
  # PyTerrier BM25
  python scripts/01_index.py --dataset nq --corpus_path /data/beir/nq --indexes pyterrier
  
  # Multiple indexes
  python scripts/01_index.py --dataset nq --indexes bge,splade
"""
    )
    parser.add_argument("--dataset", default="nq", choices=["nq", "hotpotqa"],
                        help="Dataset name")
    parser.add_argument("--corpus_path", help="Path to BEIR corpus (for PyTerrier)")
    parser.add_argument("--index_path", help="Output path for PyTerrier index")
    parser.add_argument("--indexes", default="bge",
                        help="Comma-separated: pyterrier,bge,splade,contriever")
    parser.add_argument("--hnsw", action="store_true",
                        help="Build HNSW segments for BGE (fast dense search)")
    parser.add_argument("--hnsw-segments", type=int, default=4,
                        help="Number of HNSW segments")
    parser.add_argument("--hnsw-threads", type=int, default=8,
                        help="Threads for HNSW build")
    parser.add_argument("--limit", type=int, help="Limit corpus size (PyTerrier)")
    parser.add_argument("--force", action="store_true", help="Force rebuild")
    parser.add_argument("--list", action="store_true", help="List available indexes")
    
    args = parser.parse_args()
    
    if args.list:
        print("\n=== Available Indexes ===\n")
        print("PyTerrier (requires corpus):")
        print("  pyterrier - BM25 inverted index\n")
        print("Pyserini Pre-built (auto-download):")
        for key, cfg in PREBUILT_INDEXES.items():
            print(f"  {key} - {cfg['description']}")
        print("\nHNSW (add --hnsw flag with bge):")
        print("  Builds segmented HNSW for fast BGE retrieval (~6ms/query)")
        return
    
    print(f"[Index] Dataset: {args.dataset}")
    print(f"[Index] Cache: {CACHE_ROOT}")
    
    requested = [idx.strip().lower() for idx in args.indexes.split(",")]
    results = {}
    
    # PyTerrier
    if "pyterrier" in requested:
        if not args.corpus_path:
            print("[Index] ERROR: --corpus_path required for pyterrier")
            results["pyterrier"] = "ERROR"
        else:
            output_dir = PROJECT_ROOT / "data" / args.dataset
            index_path = args.index_path or str(output_dir / "index" / "pyterrier")
            
            if os.path.exists(index_path) and os.listdir(index_path) and not args.force:
                print(f"[Index] PyTerrier exists: {index_path}")
                results["pyterrier"] = "EXISTS"
            else:
                build_pyterrier_index(args.corpus_path, index_path, args.limit)
                results["pyterrier"] = "BUILT"
    
    # Check if HNSW exists (skip BGE download if so)
    from src.indexing import HNSW_DATASETS
    cache_dir = Path(os.environ.get("PYSERINI_CACHE", "cache/pyserini"))
    hnsw_meta = cache_dir / "indexes" / HNSW_DATASETS.get(args.dataset, "") / "hnsw_segments_meta.json"
    hnsw_exists = hnsw_meta.exists()
    
    # Pre-built indexes
    for idx in requested:
        if idx == "pyterrier":
            continue
        # Skip BGE FAISS download if HNSW already built
        if idx == "bge" and hnsw_exists and args.hnsw:
            print(f"[Index] Skipping BGE FAISS (HNSW already exists)")
            results[idx] = "SKIPPED"
            continue
        if idx in PREBUILT_INDEXES:
            success = download_prebuilt_index(idx, args.dataset)
            results[idx] = "READY" if success else "FAILED"
        else:
            print(f"[Index] Unknown: {idx}")
            results[idx] = "UNKNOWN"
    
    # HNSW for BGE
    if args.hnsw:
        from src.indexing import build_hnsw_index
        
        if hnsw_exists:
            print(f"\n[Index] HNSW already exists")
            results["hnsw"] = "EXISTS"
        else:
            # Need BGE FAISS first
            if "bge" not in requested and results.get("bge") not in ["READY", "EXISTS"]:
                print("[Index] Downloading BGE FAISS for HNSW build...")
                success = download_prebuilt_index("bge", args.dataset)
                results["bge"] = "READY" if success else "FAILED"
            
            if results.get("bge") in ["READY", "EXISTS", None]:
                print(f"\n[Index] Building HNSW segments...")
                try:
                    build_hnsw_index(
                        dataset=args.dataset,
                        n_segments=args.hnsw_segments,
                        num_threads=args.hnsw_threads
                    )
                    results["hnsw"] = "BUILT"
                except Exception as e:
                    print(f"[Index] HNSW ERROR: {e}")
                    results["hnsw"] = "FAILED"
    
    # Summary
    print(f"\n{'='*40}")
    print("Results:")
    for idx, status in results.items():
        symbol = "✅" if status in ["READY", "BUILT", "EXISTS", "SKIPPED"] else "❌"
        print(f"  {symbol} {idx}: {status}")
    
    if "hnsw" in results and results["hnsw"] == "BUILT":
        print(f"\n💡 Use BGERetriever for ~6ms/query dense search")


if __name__ == "__main__":
    main()

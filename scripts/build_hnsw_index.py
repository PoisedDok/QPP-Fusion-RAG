#!/usr/bin/env python3
"""
Incoming: FAISS flat index --- {5.2M vectors, 768d}
Processing: Two-phase HNSW build --- {2 jobs: extraction, indexing}
Outgoing: HNSW index file --- {index_hnsw.bin}

Build HNSW index from Pyserini pre-built FAISS flat index.
Memory-efficient: extracts to mmap, releases FAISS, then builds HNSW.
"""
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import argparse
import time
import gc
import numpy as np
import faiss
import hnswlib
from pathlib import Path


def build_hnsw_index(dataset: str = "hotpotqa", ef_construction: int = 200, M: int = 32):
    """Build HNSW index from FAISS flat index (memory-efficient two-phase)."""
    
    cache_dir = Path(os.environ.get("PYSERINI_CACHE", "cache/pyserini"))
    
    INDEX_HASHES = {
        "nq": "faiss-flat.beir-v1.0.0-nq.bge-base-en-v1.5.20240107.b738bbbe7ca36532f25189b776d4e153",
        "hotpotqa": "faiss-flat.beir-v1.0.0-hotpotqa.bge-base-en-v1.5.20240107.d2c08665e8cd750bd06ceb7d23897c94"
    }
    
    index_hash = INDEX_HASHES[dataset]
    index_dir = cache_dir / "indexes" / index_hash
    
    # Output paths
    hnsw_path = index_dir / "index_hnsw.bin"
    vectors_mmap_path = index_dir / "vectors_extracted.npy"
    
    if hnsw_path.exists():
        print(f"HNSW index already exists: {hnsw_path}")
        print("Delete it to rebuild.")
        return str(hnsw_path)
    
    d = 768  # BGE embedding dimension
    
    # ========================================
    # PHASE 1: Extract vectors to disk (mmap)
    # ========================================
    if not vectors_mmap_path.exists():
        print(f"=== PHASE 1: Extract vectors to disk ===")
        print(f"Loading FAISS flat index for {dataset}...")
        t0 = time.time()
        flat_index = faiss.read_index(str(index_dir / "index"))
        n_vectors = flat_index.ntotal
        print(f"Loaded {n_vectors:,} vectors in {time.time()-t0:.1f}s")
        print(f"FAISS index memory: ~{n_vectors * d * 4 / 1e9:.1f} GB")
        
        # Create memory-mapped array for extraction
        print(f"\nCreating mmap file: {vectors_mmap_path}")
        vectors_mmap = np.memmap(
            str(vectors_mmap_path), 
            dtype=np.float32, 
            mode='w+', 
            shape=(n_vectors, d)
        )
        
        # Extract in chunks (streaming to disk)
        chunk_size = 50000  # Smaller chunks = less RAM
        print(f"Extracting {n_vectors:,} vectors in chunks of {chunk_size:,}...")
        
        t0 = time.time()
        for start in range(0, n_vectors, chunk_size):
            end = min(start + chunk_size, n_vectors)
            
            # Extract chunk directly to mmap
            for i, idx in enumerate(range(start, end)):
                vectors_mmap[idx] = flat_index.reconstruct(idx)
            
            # Flush to disk periodically
            if (end // chunk_size) % 10 == 0:
                vectors_mmap.flush()
            
            progress = end / n_vectors * 100
            elapsed = time.time() - t0
            eta = (elapsed / (end / n_vectors) - elapsed) if end > 0 else 0
            print(f"  [{progress:5.1f}%] {end:,}/{n_vectors:,} | ETA: {eta/60:.1f}min")
        
        # Final flush
        vectors_mmap.flush()
        del vectors_mmap
        
        # CRITICAL: Release FAISS index memory before Phase 2
        print(f"\nReleasing FAISS index from memory...")
        del flat_index
        gc.collect()
        
        print(f"Phase 1 complete. Vectors saved to {vectors_mmap_path}")
        print(f"File size: {vectors_mmap_path.stat().st_size / 1e9:.2f} GB")
    else:
        print(f"=== PHASE 1: SKIP (vectors already extracted) ===")
        print(f"Using existing: {vectors_mmap_path}")
        # Get n_vectors from file size
        file_size = vectors_mmap_path.stat().st_size
        n_vectors = file_size // (d * 4)
        print(f"Vectors: {n_vectors:,}")
    
    # ========================================
    # PHASE 2: Build HNSW from mmap'd vectors
    # ========================================
    print(f"\n=== PHASE 2: Build HNSW index ===")
    print(f"Memory state: FAISS released, only mmap + HNSW in RAM")
    
    # Load vectors as read-only mmap (minimal RAM footprint)
    vectors_mmap = np.memmap(
        str(vectors_mmap_path), 
        dtype=np.float32, 
        mode='r', 
        shape=(n_vectors, d)
    )
    
    # Initialize HNSW
    print(f"Initializing HNSW (M={M}, ef_construction={ef_construction})...")
    print(f"Expected HNSW memory: ~{(n_vectors * d * 4 + n_vectors * M * 2 * 8) / 1e9:.1f} GB")
    
    hnsw = hnswlib.Index(space='ip', dim=d)  # inner product for normalized vectors
    hnsw.init_index(max_elements=n_vectors, ef_construction=ef_construction, M=M)
    hnsw.set_num_threads(8)
    
    # Add vectors in chunks from mmap
    chunk_size = 100000
    total_time = 0
    
    print(f"Adding {n_vectors:,} vectors to HNSW...")
    for start in range(0, n_vectors, chunk_size):
        end = min(start + chunk_size, n_vectors)
        
        t0 = time.time()
        # Read chunk from mmap (OS handles caching efficiently)
        chunk = np.array(vectors_mmap[start:end])  # Copy to contiguous array
        
        # Add to HNSW
        hnsw.add_items(chunk, list(range(start, end)))
        
        chunk_time = time.time() - t0
        total_time += chunk_time
        
        # Free chunk immediately
        del chunk
        
        progress = end / n_vectors * 100
        eta = (total_time / (end / n_vectors) - total_time) if end > 0 else 0
        rate = (end - start) / chunk_time if chunk_time > 0 else 0
        print(f"  [{progress:5.1f}%] {end:,}/{n_vectors:,} | {rate:.0f} vec/s | ETA: {eta/60:.1f}min")
    
    # Release mmap
    del vectors_mmap
    gc.collect()
    
    # Save HNSW
    print(f"\nSaving HNSW index to {hnsw_path}...")
    hnsw.save_index(str(hnsw_path))
    print(f"Saved ({hnsw_path.stat().st_size / 1e9:.2f} GB)")
    
    # Clean up temp vectors file
    print(f"Cleaning up extracted vectors file...")
    vectors_mmap_path.unlink()
    
    # Quick test
    print(f"\nTesting search speed...")
    hnsw.set_ef(128)
    test_query = np.random.randn(1, d).astype(np.float32)
    test_query = test_query / np.linalg.norm(test_query)
    
    t0 = time.time()
    for _ in range(100):
        labels, distances = hnsw.knn_query(test_query, k=100)
    avg_time = (time.time() - t0) / 100 * 1000
    print(f"Average search time: {avg_time:.2f}ms per query")
    
    return str(hnsw_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="hotpotqa", choices=["nq", "hotpotqa"])
    parser.add_argument("--ef_construction", type=int, default=200)
    parser.add_argument("--M", type=int, default=32)
    args = parser.parse_args()
    
    build_hnsw_index(args.dataset, args.ef_construction, args.M)

"""
Incoming: FAISS flat index --- {N vectors, 768d}
Processing: Segmented HNSW build --- {2 phases: extraction, segment building}
Outgoing: HNSW segment files --- {index_hnsw_seg_*.bin, hnsw_segments_meta.json}

Build segmented HNSW index from Pyserini pre-built FAISS flat index.
- Memory-efficient: builds in segments (~4GB each vs 14GB+ monolithic)
- Resumable: skips already-built segments on restart
"""
import gc
import json
import os
import sys
import time
from pathlib import Path
from typing import List, Optional

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

# Dataset FAISS index hashes (Pyserini pre-built BGE)
HNSW_DATASETS = {
    "nq": "faiss-flat.beir-v1.0.0-nq.bge-base-en-v1.5.20240107.b738bbbe7ca36532f25189b776d4e153",
    "hotpotqa": "faiss-flat.beir-v1.0.0-hotpotqa.bge-base-en-v1.5.20240107.d2c08665e8cd750bd06ceb7d23897c94"
}

EMBEDDING_DIM = 768


def _extract_vectors(index_dir: Path, faiss_path: Path) -> tuple:
    """Extract vectors from FAISS to memory-mapped file."""
    import faiss
    import numpy as np
    
    vectors_path = index_dir / "vectors_extracted.npy"
    
    if vectors_path.exists():
        n_vectors = vectors_path.stat().st_size // (EMBEDDING_DIM * 4)
        print(f"  [HNSW] Using existing vectors ({n_vectors:,})")
        return vectors_path, n_vectors
    
    print(f"  [HNSW] Loading FAISS index...")
    t0 = time.time()
    flat_index = faiss.read_index(str(faiss_path))
    n_vectors = flat_index.ntotal
    print(f"  [HNSW] Loaded {n_vectors:,} vectors ({time.time()-t0:.1f}s)")
    
    print(f"  [HNSW] Extracting to mmap...")
    vectors_mmap = np.memmap(str(vectors_path), dtype=np.float32, mode='w+',
                             shape=(n_vectors, EMBEDDING_DIM))
    
    chunk_size = 50000
    t0 = time.time()
    for start in range(0, n_vectors, chunk_size):
        end = min(start + chunk_size, n_vectors)
        for idx in range(start, end):
            vectors_mmap[idx] = flat_index.reconstruct(idx)
        if (end // chunk_size) % 10 == 0:
            vectors_mmap.flush()
        progress = end / n_vectors * 100
        elapsed = time.time() - t0
        eta = (elapsed / (end / n_vectors) - elapsed) if end > 0 else 0
        print(f"    [{progress:5.1f}%] {end:,}/{n_vectors:,} | ETA: {eta/60:.1f}min")
    
    vectors_mmap.flush()
    del vectors_mmap, flat_index
    gc.collect()
    
    return vectors_path, n_vectors


def build_hnsw_index(
    dataset: str,
    n_segments: int = 4,
    ef_construction: int = 200,
    M: int = 16,
    num_threads: int = 8,
    cache_dir: Optional[Path] = None
) -> List[str]:
    """
    Build segmented HNSW index for fast dense retrieval.
    
    Args:
        dataset: Dataset name (nq, hotpotqa)
        n_segments: Number of segments (4 = ~1.3M vectors each)
        ef_construction: Build quality (200 = high)
        M: Connections per node (16 = balanced)
        num_threads: Build parallelism
        cache_dir: Override cache directory
    
    Returns:
        List of segment file paths
    """
    import hnswlib
    import numpy as np
    
    sys.stdout.reconfigure(line_buffering=True)
    
    if cache_dir is None:
        cache_dir = Path(os.environ.get("PYSERINI_CACHE", "cache/pyserini"))
    
    if dataset not in HNSW_DATASETS:
        raise ValueError(f"Unknown dataset: {dataset}. Available: {list(HNSW_DATASETS.keys())}")
    
    index_dir = cache_dir / "indexes" / HNSW_DATASETS[dataset]
    faiss_path = index_dir / "index"
    metadata_path = index_dir / "hnsw_segments_meta.json"
    
    if not index_dir.exists():
        raise FileNotFoundError(
            f"FAISS index not found: {index_dir}\n"
            f"Download first with: python scripts/01_index.py --dataset {dataset} --indexes bge"
        )
    
    # Check if complete
    if metadata_path.exists():
        with open(metadata_path) as f:
            meta = json.load(f)
        if all((index_dir / s["path"]).exists() for s in meta["segments"]):
            print(f"  [HNSW] Already complete: {len(meta['segments'])} segments")
            return [str(index_dir / s["path"]) for s in meta["segments"]]
    
    print(f"  [HNSW] Building: segments={n_segments}, M={M}, ef={ef_construction}")
    
    # Extract vectors
    vectors_path, n_vectors = _extract_vectors(index_dir, faiss_path)
    
    # Build segments
    segment_size = (n_vectors + n_segments - 1) // n_segments
    mem_gb = (segment_size * EMBEDDING_DIM * 4 + segment_size * M * 2 * 8) / 1e9
    print(f"  [HNSW] ~{segment_size:,} vectors/segment, ~{mem_gb:.1f}GB peak memory")
    
    vectors_mmap = np.memmap(str(vectors_path), dtype=np.float32, mode='r',
                             shape=(n_vectors, EMBEDDING_DIM))
    
    segment_paths = []
    for seg_idx in range(n_segments):
        seg_path = index_dir / f"index_hnsw_seg_{seg_idx}.bin"
        segment_paths.append(str(seg_path))
        
        if seg_path.exists():
            print(f"  [HNSW] Segment {seg_idx} exists, skipping")
            continue
        
        start = seg_idx * segment_size
        end = min((seg_idx + 1) * segment_size, n_vectors)
        seg_n = end - start
        
        print(f"  [HNSW] Building segment {seg_idx}/{n_segments-1} ({seg_n:,} vectors)...")
        
        hnsw = hnswlib.Index(space='ip', dim=EMBEDDING_DIM)
        hnsw.init_index(max_elements=seg_n, ef_construction=ef_construction, M=M)
        hnsw.set_num_threads(num_threads)
        
        t0 = time.time()
        seg_vectors = np.array(vectors_mmap[start:end])
        hnsw.add_items(seg_vectors, list(range(seg_n)))
        build_time = time.time() - t0
        
        del seg_vectors
        gc.collect()
        
        hnsw.save_index(str(seg_path))
        print(f"    Built in {build_time:.1f}s ({seg_n/build_time:.0f} vec/s), "
              f"saved {seg_path.stat().st_size/1e9:.2f}GB")
        
        del hnsw
        gc.collect()
    
    del vectors_mmap
    gc.collect()
    
    # Save metadata
    metadata = {
        "n_segments": n_segments,
        "n_vectors": n_vectors,
        "segment_size": segment_size,
        "M": M,
        "ef_construction": ef_construction,
        "segments": [
            {"index": i, "start_global_id": i * segment_size,
             "end_global_id": min((i+1) * segment_size, n_vectors),
             "path": f"index_hnsw_seg_{i}.bin"}
            for i in range(n_segments)
        ]
    }
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    # Cleanup temp vectors
    if vectors_path.exists():
        print(f"  [HNSW] Cleaning up temp vectors ({vectors_path.stat().st_size/1e9:.1f}GB)")
        vectors_path.unlink()
    
    total_gb = sum(Path(p).stat().st_size for p in segment_paths) / 1e9
    print(f"  [HNSW] Complete: {n_segments} segments, {total_gb:.2f}GB total")
    
    return segment_paths

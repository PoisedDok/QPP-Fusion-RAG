#!/usr/bin/env python3
"""
Incoming: .res run files --- {TREC format}
Processing: QPP computation --- {13 QPP methods per run}
Outgoing: .qpp files --- {per-query QPP scores}

Step 3: Compute QPP Scores
--------------------------
Computes 13 QPP methods for each retriever run.

Usage:
    python scripts/03_qpp.py
    python scripts/03_qpp.py --runs_dir data/nq/runs --qpp_dir data/nq/qpp
"""

import os
import sys
import argparse
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.qpp import compute_qpp_for_res_file


def main():
    parser = argparse.ArgumentParser(description="Step 3: Compute QPP Scores")
    parser.add_argument("--runs_dir", default=None, help="Directory with .res files")
    parser.add_argument("--qpp_dir", default=None, help="Output directory for QPP files")
    parser.add_argument("--queries", default=None, help="Path to queries.jsonl (BEIR format)")
    parser.add_argument("--top_k", type=int, default=100, help="Top-k for QPP computation")
    parser.add_argument("--normalize", default="minmax", choices=["none", "minmax", "zscore"])
    parser.add_argument("--force", action="store_true", help="Recompute even if QPP file exists")
    args = parser.parse_args()
    
    # Setup paths
    output_dir = PROJECT_ROOT / "data" / "nq"
    runs_dir = Path(args.runs_dir) if args.runs_dir else output_dir / "runs"
    qpp_dir = Path(args.qpp_dir) if args.qpp_dir else output_dir / "qpp"
    
    # Auto-detect queries.jsonl if not specified
    queries_path = args.queries
    if not queries_path:
        # Try common locations
        for candidate in [
            output_dir / "BEIR-nq" / "queries.jsonl",
            output_dir / "queries.jsonl",
            PROJECT_ROOT / "data" / "nq" / "BEIR-nq" / "queries.jsonl"
        ]:
            if candidate.exists():
                queries_path = str(candidate)
                break
    
    if queries_path:
        print(f"[03_qpp] Using query texts from: {queries_path}")
    else:
        print(f"[03_qpp] WARNING: No queries.jsonl found - IDF-based methods will be inaccurate")
    
    os.makedirs(qpp_dir, exist_ok=True)
    
    # Find .res files (not .norm.res)
    res_files = [f for f in os.listdir(runs_dir) if f.endswith(".res") and not f.endswith(".norm.res")]
    
    if not res_files:
        print(f"[03_qpp] No .res files found in {runs_dir}")
        return
    
    print(f"[03_qpp] Processing {len(res_files)} run files")
    print(f"[03_qpp] QPP methods: 13 (NQC, RSD, WIG, SMV, UEF, ...)")
    
    # Process files sequentially (each file uses optimized batch QPP internally)
    processed = 0
    skipped = 0
    for res_file in res_files:
        res_path = runs_dir / res_file
        qpp_output = qpp_dir / res_file.replace(".res", ".res.mmnorm.qpp")
        
        if qpp_output.exists() and not args.force:
            print(f"[03_qpp] SKIP {res_file} (already exists, use --force to recompute)")
            skipped += 1
            continue
        
        print(f"\n[03_qpp] Processing {res_file}...")
        compute_qpp_for_res_file(
            str(res_path),
            str(qpp_output),
            top_k=args.top_k,
            normalize=args.normalize,
            queries_path=queries_path
        )
        processed += 1
    
    print(f"\n[03_qpp] Processed: {processed}, Skipped: {skipped}")
    
    print(f"\n=== Step 3 Complete ===")
    print(f"QPP files: {qpp_dir}")
    print(f"Files: {list(qpp_dir.glob('*.qpp'))}")


if __name__ == "__main__":
    main()


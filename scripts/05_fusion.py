#!/usr/bin/env python3
"""
Incoming: .norm.res files, .qpp files, trained models --- {runs, QPP, models}
Processing: fusion --- {9 methods: CombSUM, CombMNZ, RRF + weighted + learned}
Outgoing: fused .res files --- {TREC format}

Step 5: Apply Fusion
--------------------
Applies fusion strategies to combine multiple retriever runs.

Methods:
  combsum   - Sum of normalized scores
  combmnz   - CombSUM × number of rankers returning doc  
  rrf       - Reciprocal Rank Fusion
  wcombsum  - QPP-weighted CombSUM
  wcombmnz  - QPP-weighted CombMNZ
  wrrf      - QPP-weighted RRF
  learned   - ML model learned weights
  all       - Run all methods and output comparison

Usage:
    python scripts/04_fusion.py --method combsum
    python scripts/04_fusion.py --method wcombsum --qpp_model RSD
    python scripts/04_fusion.py --method learned --model_path data/nq/models/fusion_weights_model.pkl
    python scripts/04_fusion.py --method all  # Run all methods
"""

import os
import sys
import argparse
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.fusion import (
    run_fusion,
    load_runs,
    load_qpp_scores,
    combsum,
    combmnz,
    rrf,
    weighted_combsum,
    weighted_combmnz,
    weighted_rrf,
    learned_fusion,
    write_runfile,
    get_qpp_index
)


METHODS = ["combsum", "combmnz", "rrf", "wcombsum", "wcombmnz", "wrrf", "learned"]


def run_all_methods(
    runs_dir: Path,
    qpp_dir: Path,
    fused_dir: Path,
    qpp_model: str = "RSD",
    model_path: str = None,
    rrf_k: int = 60
):
    """Run all fusion methods and output comparison."""
    print(f"[04_fusion] Running all fusion methods...")
    print(f"[04_fusion] QPP model for weighted: {qpp_model}")
    
    results = {}
    
    # Load data once
    runs = load_runs(str(runs_dir), use_normalized=True)
    print(f"[04_fusion] Loaded {len(runs)} rankers: {list(runs.keys())}")
    
    # Unweighted methods
    print("\n--- Unweighted Methods ---")
    
    # CombSUM
    print("[04_fusion] Running CombSUM...")
    fused = combsum(runs)
    output_path = fused_dir / "combsum.res"
    write_runfile(fused, str(output_path), "combsum")
    results["combsum"] = output_path
    
    # CombMNZ
    print("[04_fusion] Running CombMNZ...")
    fused = combmnz(runs)
    output_path = fused_dir / "combmnz.res"
    write_runfile(fused, str(output_path), "combmnz")
    results["combmnz"] = output_path
    
    # RRF
    print(f"[04_fusion] Running RRF (k={rrf_k})...")
    fused = rrf(runs, k=rrf_k)
    output_path = fused_dir / f"rrf_k{rrf_k}.res"
    write_runfile(fused, str(output_path), f"rrf-k{rrf_k}")
    results["rrf"] = output_path
    
    # Weighted methods (need QPP)
    if qpp_dir.exists():
        print("\n--- QPP-Weighted Methods ---")
        qpp_data = load_qpp_scores(str(qpp_dir))
        qpp_index = get_qpp_index(qpp_model)
        
        # W-CombSUM
        print(f"[04_fusion] Running W-CombSUM ({qpp_model})...")
        fused = weighted_combsum(runs, qpp_data, qpp_index)
        output_path = fused_dir / f"wcombsum_{qpp_model.lower()}.res"
        write_runfile(fused, str(output_path), f"wcombsum-{qpp_model.lower()}")
        results["wcombsum"] = output_path
        
        # W-CombMNZ
        print(f"[04_fusion] Running W-CombMNZ ({qpp_model})...")
        fused = weighted_combmnz(runs, qpp_data, qpp_index)
        output_path = fused_dir / f"wcombmnz_{qpp_model.lower()}.res"
        write_runfile(fused, str(output_path), f"wcombmnz-{qpp_model.lower()}")
        results["wcombmnz"] = output_path
        
        # W-RRF
        print(f"[04_fusion] Running W-RRF ({qpp_model})...")
        fused = weighted_rrf(runs, qpp_data, qpp_index, k=rrf_k)
        output_path = fused_dir / f"wrrf_{qpp_model.lower()}.res"
        write_runfile(fused, str(output_path), f"wrrf-{qpp_model.lower()}")
        results["wrrf"] = output_path
        
        # Learned fusion (if model exists)
        if model_path and Path(model_path).exists():
            print("[04_fusion] Running Learned Fusion...")
            fused = learned_fusion(runs, qpp_data, model_path)
            output_path = fused_dir / "learned.res"
            write_runfile(fused, str(output_path), "learned")
            results["learned"] = output_path
        else:
            print("[04_fusion] Skipping learned fusion (no model)")
    else:
        print(f"[04_fusion] Skipping weighted methods (no QPP at {qpp_dir})")
    
    print(f"\n=== Generated {len(results)} fusion runs ===")
    for method, path in results.items():
        print(f"  {method}: {path}")
    
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Step 4: Multi-Method Fusion",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Methods:
  combsum   - Sum of normalized scores
  combmnz   - CombSUM × number of rankers returning doc
  rrf       - Reciprocal Rank Fusion
  wcombsum  - QPP-weighted CombSUM
  wcombmnz  - QPP-weighted CombMNZ
  wrrf      - QPP-weighted RRF
  learned   - ML model learned weights
  all       - Run all methods

Examples:
  python scripts/04_fusion.py --method combsum
  python scripts/04_fusion.py --method wcombsum --qpp_model NQC
  python scripts/04_fusion.py --method all
"""
    )
    parser.add_argument("--method", default="wcombsum",
                        choices=METHODS + ["all"],
                        help="Fusion method (default: wcombsum)")
    parser.add_argument("--runs_dir", default=None, help="Directory with .norm.res files")
    parser.add_argument("--qpp_dir", default=None, help="Directory with .qpp files")
    parser.add_argument("--qpp_model", default="RSD", help="QPP model for weights")
    parser.add_argument("--model_path", default=None, help="Path to learned model")
    parser.add_argument("--output", default=None, help="Output fused run file")
    parser.add_argument("--rrf_k", type=int, default=60, help="RRF k constant")
    args = parser.parse_args()
    
    # Setup paths
    output_dir = PROJECT_ROOT / "data" / "nq"
    runs_dir = Path(args.runs_dir) if args.runs_dir else output_dir / "runs"
    qpp_dir = Path(args.qpp_dir) if args.qpp_dir else output_dir / "qpp"
    fused_dir = output_dir / "fused"
    
    os.makedirs(fused_dir, exist_ok=True)
    
    # Default model path
    model_path = args.model_path or str(output_dir / "models" / "fusion_weights_model.pkl")
    
    if args.method == "all":
        # Run all methods
        run_all_methods(
            runs_dir=runs_dir,
            qpp_dir=qpp_dir,
            fused_dir=fused_dir,
            qpp_model=args.qpp_model,
            model_path=model_path,
            rrf_k=args.rrf_k
        )
    else:
        # Run single method
        if args.output:
            output_path = args.output
        else:
            if args.method.startswith("w"):
                output_path = str(fused_dir / f"{args.method}_{args.qpp_model.lower()}.res")
            elif args.method == "learned":
                output_path = str(fused_dir / "learned.res")
            else:
                output_path = str(fused_dir / f"{args.method}.res")
        
        print(f"[04_fusion] Running {args.method}...")
        
        run_fusion(
            method=args.method,
            runs_dir=str(runs_dir),
            qpp_dir=str(qpp_dir) if qpp_dir.exists() else None,
            qpp_model=args.qpp_model,
            model_path=model_path if args.method == "learned" else None,
            output_path=output_path,
            rrf_k=args.rrf_k
        )
        
        print(f"\n=== Step 4 Complete ===")
        print(f"Fused run: {output_path}")


if __name__ == "__main__":
    main()

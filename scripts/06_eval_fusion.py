#!/usr/bin/env python3
"""
Incoming: fused .res files, qrels --- {TREC runs, relevance judgments}
Processing: IR evaluation --- {ir_measures: NDCG@10, MRR@10, Recall@10}
Outgoing: comparison results --- {JSON + table}

Step 6: Evaluate Fusion (IR Metrics)
------------------------------------
Evaluates all fusion methods using ir_measures and generates comparison table.

Evaluates:
  - Unweighted: CombSUM, CombMNZ, RRF (via ranx)
  - QPP-Weighted: W-CombSUM, W-CombMNZ, W-RRF
  - Learned: PerRetriever, MultiOutput, MLP

STRICT: Uses ir_measures for evaluation. No manual implementations.

Usage:
    python scripts/06_eval_fusion.py
    python scripts/06_eval_fusion.py --qpp_model NQC
"""

import os
import sys
import json
import argparse
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# STRICT: Use research-grade packages
from src.fusion import (
    load_runs, load_qpp_scores, get_qpp_index,
    combsum, combmnz, rrf,
    weighted_combsum, weighted_combmnz, weighted_rrf,
    learned_fusion, write_runfile
)
from src.evaluation import IREvaluator


def load_qrels(qrels_path: Path) -> Dict[str, Dict[str, int]]:
    """Load qrels."""
    qrels = defaultdict(dict)
    with open(qrels_path) as f:
        next(f)
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) >= 3:
                qrels[parts[0]][parts[1]] = int(parts[2])
    return dict(qrels)


def evaluate_fusion(
    fused: Dict[str, List[Tuple[str, float]]],
    qrels: Dict[str, Dict[str, int]],
    evaluator: IREvaluator
) -> Dict[str, float]:
    """Evaluate fusion results using ir_measures."""
    metrics = evaluator.evaluate(fused, qrels, per_query=False)
    metrics['n_queries'] = len([q for q in fused.keys() if q in qrels])
    return metrics


def run_comparison(
    runs_dir: Path,
    qpp_dir: Path,
    qrels_path: Path,
    models_dir: Path,
    output_dir: Path,
    qpp_model: str = "RSD"
):
    """Run all fusion methods and compare."""
    print("="*70)
    print("FUSION METHOD COMPARISON (ir_measures)")
    print("="*70)
    
    # Load data
    runs = load_runs(str(runs_dir), use_normalized=True)
    qrels = load_qrels(qrels_path)
    
    print(f"Retrievers: {list(runs.keys())}")
    print(f"Queries with qrels: {len(qrels)}")
    
    # Initialize evaluator with desired metrics
    evaluator = IREvaluator(metrics=["nDCG@10", "RR@10", "R@10"])
    
    results = []
    
    # === Unweighted Methods ===
    print("\n--- Unweighted Methods (ranx) ---")
    
    # CombSUM
    print("Running CombSUM...")
    fused = combsum(runs)
    metrics = evaluate_fusion(fused, qrels, evaluator)
    results.append({'method': 'CombSUM', 'type': 'unweighted', **metrics})
    write_runfile(fused, str(output_dir / "combsum.res"), "combsum")
    
    # CombMNZ
    print("Running CombMNZ...")
    fused = combmnz(runs)
    metrics = evaluate_fusion(fused, qrels, evaluator)
    results.append({'method': 'CombMNZ', 'type': 'unweighted', **metrics})
    write_runfile(fused, str(output_dir / "combmnz.res"), "combmnz")
    
    # RRF
    print("Running RRF...")
    fused = rrf(runs)
    metrics = evaluate_fusion(fused, qrels, evaluator)
    results.append({'method': 'RRF', 'type': 'unweighted', **metrics})
    write_runfile(fused, str(output_dir / "rrf.res"), "rrf")
    
    # === QPP-Weighted Methods ===
    if qpp_dir.exists():
        print(f"\n--- QPP-Weighted Methods ({qpp_model}) ---")
        qpp_data = load_qpp_scores(str(qpp_dir))
        qpp_index = get_qpp_index(qpp_model)
        
        # W-CombSUM
        print(f"Running W-CombSUM ({qpp_model})...")
        fused = weighted_combsum(runs, qpp_data, qpp_index)
        metrics = evaluate_fusion(fused, qrels, evaluator)
        results.append({'method': f'W-CombSUM ({qpp_model})', 'type': 'qpp-weighted', **metrics})
        write_runfile(fused, str(output_dir / f"wcombsum_{qpp_model.lower()}.res"), f"wcombsum-{qpp_model.lower()}")
        
        # W-CombMNZ
        print(f"Running W-CombMNZ ({qpp_model})...")
        fused = weighted_combmnz(runs, qpp_data, qpp_index)
        metrics = evaluate_fusion(fused, qrels, evaluator)
        results.append({'method': f'W-CombMNZ ({qpp_model})', 'type': 'qpp-weighted', **metrics})
        write_runfile(fused, str(output_dir / f"wcombmnz_{qpp_model.lower()}.res"), f"wcombmnz-{qpp_model.lower()}")
        
        # W-RRF
        print(f"Running W-RRF ({qpp_model})...")
        fused = weighted_rrf(runs, qpp_data, qpp_index)
        metrics = evaluate_fusion(fused, qrels, evaluator)
        results.append({'method': f'W-RRF ({qpp_model})', 'type': 'qpp-weighted', **metrics})
        write_runfile(fused, str(output_dir / f"wrrf_{qpp_model.lower()}.res"), f"wrrf-{qpp_model.lower()}")
        
        # === Learned Methods ===
        print("\n--- Learned Methods ---")
        
        for model_name in ["per_retriever", "multioutput", "mlp"]:
            model_path = models_dir / f"fusion_{model_name}.pkl"
            if model_path.exists():
                print(f"Running Learned ({model_name})...")
                fused = learned_fusion(runs, qpp_data, str(model_path))
                metrics = evaluate_fusion(fused, qrels, evaluator)
                results.append({'method': f'Learned ({model_name})', 'type': 'learned', **metrics})
                write_runfile(fused, str(output_dir / f"learned_{model_name}.res"), f"learned-{model_name}")
            else:
                print(f"  Skipping {model_name} (model not found)")
    
    # === Results Table ===
    print("\n" + "="*70)
    print("RESULTS")
    print("="*70)
    
    # Sort by NDCG
    results.sort(key=lambda x: -x.get('nDCG@10', 0))
    
    # Get baseline for improvement calculation
    baseline_ndcg = next((r.get('nDCG@10', 0) for r in results if r['method'] == 'CombSUM'), results[-1].get('nDCG@10', 0))
    
    print(f"\n{'Method':<25} {'Type':<15} {'NDCG@10':<10} {'MRR@10':<10} {'Recall@10':<10} {'Δ NDCG':<10}")
    print("-"*80)
    
    for r in results:
        ndcg = r.get('nDCG@10', 0)
        mrr = r.get('RR@10', 0)
        recall = r.get('R@10', 0)
        improvement = (ndcg - baseline_ndcg) / baseline_ndcg * 100 if baseline_ndcg > 0 else 0
        sign = "+" if improvement >= 0 else ""
        print(f"{r['method']:<25} {r['type']:<15} {ndcg:.4f}     {mrr:.4f}     {recall:.4f}     {sign}{improvement:.1f}%")
    
    print("-"*80)
    
    # Save results
    results_file = output_dir / "comparison_results.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {results_file}")
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Compare Fusion Methods (ir_measures)")
    parser.add_argument("--data_dir", default=None, help="Data directory")
    parser.add_argument("--corpus_path", default=None, help="Path to BEIR dataset")
    parser.add_argument("--qpp_model", default="RSD", help="QPP model for weighted methods")
    args = parser.parse_args()
    
    # Paths
    data_dir = Path(args.data_dir) if args.data_dir else PROJECT_ROOT / "data" / "nq"
    runs_dir = data_dir / "runs"
    qpp_dir = data_dir / "qpp"
    models_dir = data_dir / "models"
    output_dir = data_dir / "fused"
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Qrels path
    if args.corpus_path:
        qrels_path = Path(args.corpus_path) / "qrels" / "test.tsv"
    else:
        qrels_path = Path("/Volumes/Disk-D/RAGit/data/beir/datasets/nq/qrels/test.tsv")
    
    run_comparison(runs_dir, qpp_dir, qrels_path, models_dir, output_dir, args.qpp_model)


if __name__ == "__main__":
    main()

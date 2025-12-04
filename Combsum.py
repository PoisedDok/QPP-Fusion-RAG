#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Incoming: data/RL/*/*.norm.res, data/Predictors/*/*.mmnorm.qpp --- {TREC run files, QPP scores}
Processing: score aggregation --- {2 jobs: normalization, fusion}
Outgoing: CLI output path --- {TREC run file}

CombSum and Weighted CombSum (W-CombSum) with QPP-based weights
---------------------------------------------------------------
CombSum(d,q)   = Σ S_i^norm(d,q)
W-CombSum(d,q) = Σ w_i(q) × S_i^norm(d,q)

Unlike CombMNZ, CombSum does NOT multiply by document count.
"""

import os
import argparse
import pandas as pd
from collections import defaultdict
from typing import Dict, Optional

# QPP model index mapping
QPP_MODELS = {
    0: "SMV", 1: "Sigma_max", 2: "Sigma(%)", 3: "NQC", 4: "UEF", 5: "RSD",
    6: "QPP-PRP", 7: "WIG", 8: "SCNQC", 9: "QV-NQC", 10: "DM",
    11: "NQA-QPP", 12: "BERTQPP"
}


def load_runs(res_path: str) -> Dict[str, pd.DataFrame]:
    """
    Load normalized run files (.norm.res) into dict of DataFrames.
    Format: qid Q0 docno rank score runid
    """
    runs = {}
    files = [f for f in os.listdir(res_path) if f.endswith(".norm.res")]
    
    if not files:
        raise FileNotFoundError(f"No .norm.res files found in {res_path}")
    
    for f in files:
        ranker = f.replace(".norm.res", "")
        df = pd.read_csv(
            os.path.join(res_path, f),
            sep=r"\s+",
            names=["qid", "iter", "docno", "rank", "score", "runid"],
            dtype={"qid": str, "docno": str}
        )
        df["qid"] = df["qid"].astype(str)
        runs[ranker] = df
    
    print(f"Loaded {len(runs)} rankers: {list(runs.keys())}")
    return runs


def load_qpp_estimates(qpp_path: str) -> Dict[str, Dict[str, list]]:
    """
    Load QPP files: {qid: {ranker: [qpp_scores...]}}
    Each score list has 13 values (one per QPP method).
    """
    qpp_data = defaultdict(dict)
    files = [f for f in os.listdir(qpp_path) if f.endswith(".mmnorm.qpp")]
    
    if not files:
        raise FileNotFoundError(f"No .mmnorm.qpp files found in {qpp_path}")
    
    for f in files:
        ranker = os.path.basename(f).replace(".res.mmnorm.qpp", "")
        filepath = os.path.join(qpp_path, f)
        
        with open(filepath, "r") as fin:
            for line in fin:
                parts = line.strip().split("\t")
                if len(parts) < 2:
                    continue
                qid = parts[0]
                scores = [float(x) for x in parts[1:]]
                qpp_data[qid][ranker] = scores
    
    print(f"Loaded QPP estimates for {len(qpp_data)} queries")
    return dict(qpp_data)


def get_qpp_index(model_name: str) -> int:
    """Resolve QPP model name to index. Returns -1 for 'fusion' (average all)."""
    if model_name.lower() == "fusion":
        return -1
    
    for idx, name in QPP_MODELS.items():
        if name.lower() == model_name.lower():
            return idx
    
    valid = list(QPP_MODELS.values()) + ["fusion"]
    raise ValueError(f"Invalid QPP model '{model_name}'. Valid: {valid}")


def combsum(runs: Dict[str, pd.DataFrame]) -> Dict[str, list]:
    """
    Standard CombSum: sum normalized scores across rankers.
    CombSum(d,q) = Σ S_i(d,q)
    """
    fused = defaultdict(list)
    all_qids = sorted(set.union(*[set(df["qid"].unique()) for df in runs.values()]))
    
    for qid in all_qids:
        doc_scores = defaultdict(float)
        
        for ranker, df in runs.items():
            sub = df[df["qid"] == qid]
            for _, row in sub.iterrows():
                doc_scores[row["docno"]] += row["score"]
        
        for docid, score in doc_scores.items():
            fused[qid].append((docid, score))
    
    return dict(fused)


def weighted_combsum(
    runs: Dict[str, pd.DataFrame],
    qpp_data: Dict[str, Dict[str, list]],
    qpp_index: int
) -> Dict[str, list]:
    """
    Weighted CombSum: sum weighted scores using QPP estimates.
    W-CombSum(d,q) = Σ w_i(q) × S_i(d,q)
    
    Args:
        runs: Dict of ranker DataFrames
        qpp_data: QPP estimates {qid: {ranker: [scores]}}
        qpp_index: QPP method index, or -1 for average of all methods
    """
    fused = defaultdict(list)
    all_qids = sorted(set.union(*[set(df["qid"].unique()) for df in runs.values()]))
    
    for qid in all_qids:
        doc_scores = defaultdict(float)
        
        for ranker, df in runs.items():
            sub = df[df["qid"] == qid]
            
            # Determine weight for this (query, ranker) pair
            if qid in qpp_data and ranker in qpp_data[qid]:
                if qpp_index == -1:
                    # Fusion mode: average all QPP methods
                    weight = sum(qpp_data[qid][ranker]) / len(QPP_MODELS)
                else:
                    weight = qpp_data[qid][ranker][qpp_index]
            else:
                weight = 1.0  # fallback
            
            # Accumulate weighted scores
            for _, row in sub.iterrows():
                doc_scores[row["docno"]] += weight * row["score"]
        
        for docid, score in doc_scores.items():
            fused[qid].append((docid, score))
    
    return dict(fused)


def write_runfile(fused: Dict[str, list], output_path: str, tag: str = "combsum"):
    """Write fused results in TREC format."""
    with open(output_path, "w") as fout:
        for qid in sorted(fused.keys(), key=lambda x: int(x) if x.isdigit() else x):
            ranked = sorted(fused[qid], key=lambda x: x[1], reverse=True)
            for rank, (docid, score) in enumerate(ranked, start=1):
                fout.write(f"{qid} Q0 {docid} {rank} {score:.6f} {tag}\n")
    
    print(f"Wrote fused run to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="CombSum / W-CombSum fusion",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Standard CombSum
  python Combsum.py --res_path data/RL/2019 --output combsum_2019.res

  # Weighted CombSum with NQC
  python Combsum.py --res_path data/RL/2019 --qpp_path data/Predictors/2019 \\
                    --qpp_model NQC --output wcombsum_nqc_2019.res

  # Weighted CombSum with fusion (average all QPP methods)
  python Combsum.py --res_path data/RL/2019 --qpp_path data/Predictors/2019 \\
                    --qpp_model fusion --output wcombsum_fusion_2019.res
"""
    )
    parser.add_argument("--res_path", required=True,
                        help="Directory with .norm.res run files")
    parser.add_argument("--qpp_path", default=None,
                        help="Directory with .mmnorm.qpp files (required for weighted)")
    parser.add_argument("--qpp_model", default=None,
                        help="QPP model name (e.g., NQC, SCNQC, WIG, fusion)")
    parser.add_argument("--output", required=True,
                        help="Output TREC run file path")
    args = parser.parse_args()

    runs = load_runs(args.res_path)

    if args.qpp_model:
        if not args.qpp_path:
            raise ValueError("--qpp_path required when using --qpp_model")
        
        qpp_data = load_qpp_estimates(args.qpp_path)
        qpp_index = get_qpp_index(args.qpp_model)
        fused = weighted_combsum(runs, qpp_data, qpp_index)
        tag = f"wcombsum-{args.qpp_model.lower()}"
    else:
        fused = combsum(runs)
        tag = "combsum"

    write_runfile(fused, args.output, tag=tag)


if __name__ == "__main__":
    main()

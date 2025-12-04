#!/usr/bin/env python3
"""
Fusion Methods for Multi-Retriever RAG
---------------------------------------
Implements all fusion baselines and QPP-weighted variants:

Unweighted:
- CombSUM: Σ S_i(d,q)
- CombMNZ: |{i: d ∈ R_i}| × Σ S_i(d,q)
- RRF: Σ 1/(k + rank_i(d,q))

QPP-Weighted:
- W-CombSUM: Σ w_i(q) × S_i(d,q)
- W-CombMNZ: |{i}| × Σ w_i(q) × S_i(d,q)
- W-RRF: Σ w_i(q) / (k + rank_i(d,q))

Learned:
- Learned fusion with ML model weights
"""

import os
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Optional, Tuple, Any


# QPP model index mapping (strict order from research plan)
QPP_MODELS = {
    0: "SMV", 1: "Sigma_max", 2: "Sigma(%)", 3: "NQC", 4: "UEF", 5: "RSD",
    6: "QPP-PRP", 7: "WIG", 8: "SCNQC", 9: "QV-NQC", 10: "DM",
    11: "NQA-QPP", 12: "BERTQPP"
}

QPP_MODEL_NAMES = list(QPP_MODELS.values())


def load_runs(res_path: str, use_normalized: bool = True) -> Dict[str, pd.DataFrame]:
    """
    Load run files into dict of DataFrames.
    
    Args:
        res_path: Directory with run files
        use_normalized: If True, load .norm.res files; else .res files
    
    Returns:
        {retriever_name: DataFrame with qid, docno, rank, score}
    """
    runs = {}
    suffix = ".norm.res" if use_normalized else ".res"
    
    files = [f for f in os.listdir(res_path) if f.endswith(suffix)]
    
    if not files:
        raise FileNotFoundError(f"No {suffix} files found in {res_path}")
    
    for f in files:
        ranker = f.replace(suffix, "")
        df = pd.read_csv(
            os.path.join(res_path, f),
            sep=r"\s+",
            names=["qid", "iter", "docno", "rank", "score", "runid"],
            dtype={"qid": str, "docno": str}
        )
        df["qid"] = df["qid"].astype(str)
        runs[ranker] = df
    
    return runs


def load_qpp_scores(qpp_path: str) -> Dict[str, Dict[str, List[float]]]:
    """
    Load QPP files: {qid: {retriever: [13 qpp_scores]}}
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
    
    return dict(qpp_data)


def get_qpp_weight(
    qid: str,
    ranker: str,
    qpp_data: Dict[str, Dict[str, List[float]]],
    qpp_index: int = 5,  # Default: RSD
    fusion_mode: bool = False
) -> float:
    """
    Get QPP weight for (query, ranker) pair.
    
    Args:
        qid: Query ID
        ranker: Retriever name
        qpp_data: QPP scores dict
        qpp_index: Which QPP method to use (0-12), or -1 for fusion
        fusion_mode: If True, average all 13 QPP methods
    
    Returns:
        Weight value (0-1 range after normalization)
    """
    if qid not in qpp_data or ranker not in qpp_data[qid]:
        return 1.0  # Fallback
    
    scores = qpp_data[qid][ranker]
    
    if fusion_mode or qpp_index == -1:
        return sum(scores) / len(scores)
    else:
        return scores[qpp_index] if qpp_index < len(scores) else 1.0


def get_qpp_index(model_name: str) -> int:
    """Resolve QPP model name to index. Returns -1 for 'fusion'."""
    if model_name.lower() == "fusion":
        return -1
    
    for idx, name in QPP_MODELS.items():
        if name.lower() == model_name.lower():
            return idx
    
    # Try case-insensitive match
    name_lower = model_name.lower()
    for idx, name in QPP_MODELS.items():
        if name.lower() == name_lower:
            return idx
    
    valid = list(QPP_MODELS.values()) + ["fusion"]
    raise ValueError(f"Invalid QPP model '{model_name}'. Valid: {valid}")


# =============================================================================
# Unweighted Fusion Methods
# =============================================================================

def combsum(runs: Dict[str, pd.DataFrame]) -> Dict[str, List[Tuple[str, float]]]:
    """
    CombSUM: Sum normalized scores across rankers.
    
    Formula: CombSUM(d,q) = Σ S_i(d,q)
    
    Returns:
        {qid: [(docid, fused_score), ...]}
    """
    # Concatenate all runs and group by (qid, docno)
    print("[fusion] Running CombSUM...")
    all_dfs = []
    for ranker, df in runs.items():
        all_dfs.append(df[["qid", "docno", "score"]])
    
    combined = pd.concat(all_dfs, ignore_index=True)
    aggregated = combined.groupby(["qid", "docno"])["score"].sum().reset_index()
    
    # Convert to dict format
    fused = defaultdict(list)
    for _, row in aggregated.iterrows():
        fused[row["qid"]].append((row["docno"], row["score"]))
    
    print(f"[fusion] CombSUM done: {len(fused)} queries")
    return dict(fused)


def combmnz(runs: Dict[str, pd.DataFrame]) -> Dict[str, List[Tuple[str, float]]]:
    """
    CombMNZ: Multiply sum by number of rankers returning the document.
    
    Formula: CombMNZ(d,q) = |{i: d ∈ R_i}| × Σ S_i(d,q)
    
    Returns:
        {qid: [(docid, fused_score), ...]}
    """
    print("[fusion] Running CombMNZ...")
    all_dfs = []
    for ranker, df in runs.items():
        all_dfs.append(df[["qid", "docno", "score"]])
    
    combined = pd.concat(all_dfs, ignore_index=True)
    
    # Group and compute sum and count
    aggregated = combined.groupby(["qid", "docno"]).agg(
        score_sum=("score", "sum"),
        count=("score", "count")
    ).reset_index()
    
    # MNZ: multiply sum by count
    aggregated["mnz_score"] = aggregated["score_sum"] * aggregated["count"]
    
    # Convert to dict format
    fused = defaultdict(list)
    for _, row in aggregated.iterrows():
        fused[row["qid"]].append((row["docno"], row["mnz_score"]))
    
    print(f"[fusion] CombMNZ done: {len(fused)} queries")
    return dict(fused)


def rrf(runs: Dict[str, pd.DataFrame], k: int = 60) -> Dict[str, List[Tuple[str, float]]]:
    """
    Reciprocal Rank Fusion (RRF).
    
    Formula: RRF(d,q) = Σ 1/(k + rank_i(d,q))
    
    Args:
        runs: Dict of ranker DataFrames
        k: RRF constant (default 60)
    
    Returns:
        {qid: [(docid, fused_score), ...]}
    """
    print("[fusion] Running RRF...")
    all_dfs = []
    for ranker, df in runs.items():
        df_copy = df[["qid", "docno", "rank"]].copy()
        df_copy["rrf_score"] = 1.0 / (k + df_copy["rank"])
        all_dfs.append(df_copy[["qid", "docno", "rrf_score"]])
    
    combined = pd.concat(all_dfs, ignore_index=True)
    aggregated = combined.groupby(["qid", "docno"])["rrf_score"].sum().reset_index()
    
    # Convert to dict format
    fused = defaultdict(list)
    for _, row in aggregated.iterrows():
        fused[row["qid"]].append((row["docno"], row["rrf_score"]))
    
    print(f"[fusion] RRF done: {len(fused)} queries")
    return dict(fused)


# =============================================================================
# QPP-Weighted Fusion Methods
# =============================================================================

def weighted_combsum(
    runs: Dict[str, pd.DataFrame],
    qpp_data: Dict[str, Dict[str, List[float]]],
    qpp_index: int = 5
) -> Dict[str, List[Tuple[str, float]]]:
    """
    QPP-Weighted CombSUM.
    
    Formula: W-CombSUM(d,q) = Σ w_i(q) × S_i(d,q)
    
    Args:
        runs: Dict of ranker DataFrames
        qpp_data: QPP scores dict
        qpp_index: QPP method index (0-12) or -1 for fusion
    
    Returns:
        {qid: [(docid, fused_score), ...]}
    """
    fused = defaultdict(list)
    all_qids = sorted(set.union(*[set(df["qid"].unique()) for df in runs.values()]))
    fusion_mode = (qpp_index == -1)
    
    for qid in all_qids:
        doc_scores = defaultdict(float)
        
        for ranker, df in runs.items():
            weight = get_qpp_weight(qid, ranker, qpp_data, qpp_index, fusion_mode)
            
            sub = df[df["qid"] == qid]
            for _, row in sub.iterrows():
                doc_scores[row["docno"]] += weight * row["score"]
        
        for docid, score in doc_scores.items():
            fused[qid].append((docid, score))
    
    return dict(fused)


def weighted_combmnz(
    runs: Dict[str, pd.DataFrame],
    qpp_data: Dict[str, Dict[str, List[float]]],
    qpp_index: int = 5
) -> Dict[str, List[Tuple[str, float]]]:
    """
    QPP-Weighted CombMNZ.
    
    Formula: W-CombMNZ(d,q) = |{i}| × Σ w_i(q) × S_i(d,q)
    
    Args:
        runs: Dict of ranker DataFrames
        qpp_data: QPP scores dict
        qpp_index: QPP method index (0-12) or -1 for fusion
    
    Returns:
        {qid: [(docid, fused_score), ...]}
    """
    fused = defaultdict(list)
    all_qids = sorted(set.union(*[set(df["qid"].unique()) for df in runs.values()]))
    fusion_mode = (qpp_index == -1)
    
    for qid in all_qids:
        doc_scores = defaultdict(float)
        doc_counts = defaultdict(int)
        
        for ranker, df in runs.items():
            weight = get_qpp_weight(qid, ranker, qpp_data, qpp_index, fusion_mode)
            
            sub = df[df["qid"] == qid]
            for _, row in sub.iterrows():
                doc_scores[row["docno"]] += weight * row["score"]
                doc_counts[row["docno"]] += 1
        
        for docid, score in doc_scores.items():
            fused[qid].append((docid, score * doc_counts[docid]))
    
    return dict(fused)


def weighted_rrf(
    runs: Dict[str, pd.DataFrame],
    qpp_data: Dict[str, Dict[str, List[float]]],
    qpp_index: int = 5,
    k: int = 60
) -> Dict[str, List[Tuple[str, float]]]:
    """
    QPP-Weighted RRF.
    
    Formula: W-RRF(d,q) = Σ w_i(q) / (k + rank_i(d,q))
    
    Args:
        runs: Dict of ranker DataFrames
        qpp_data: QPP scores dict
        qpp_index: QPP method index (0-12) or -1 for fusion
        k: RRF constant
    
    Returns:
        {qid: [(docid, fused_score), ...]}
    """
    fused = defaultdict(list)
    all_qids = sorted(set.union(*[set(df["qid"].unique()) for df in runs.values()]))
    fusion_mode = (qpp_index == -1)
    
    for qid in all_qids:
        doc_scores = defaultdict(float)
        
        for ranker, df in runs.items():
            weight = get_qpp_weight(qid, ranker, qpp_data, qpp_index, fusion_mode)
            
            sub = df[df["qid"] == qid].sort_values("rank")
            for _, row in sub.iterrows():
                doc_scores[row["docno"]] += weight / (k + row["rank"])
        
        for docid, score in doc_scores.items():
            fused[qid].append((docid, score))
    
    return dict(fused)


# =============================================================================
# Learned Fusion (ML Model Weights)
# =============================================================================

def learned_fusion(
    runs: Dict[str, pd.DataFrame],
    qpp_data: Dict[str, Dict[str, List[float]]],
    model_path: str,
    retrievers: Optional[List[str]] = None
) -> Dict[str, List[Tuple[str, float]]]:
    """
    Fusion with learned per-query weights from ML model.
    
    Args:
        runs: Dict of ranker DataFrames
        qpp_data: QPP scores dict
        model_path: Path to trained model pickle
        retrievers: List of retriever names in model order
    
    Returns:
        {qid: [(docid, fused_score), ...]}
    """
    print(f"[fusion] Running learned fusion from {model_path}...")
    
    # Load model
    with open(model_path, 'rb') as f:
        model_data = pickle.load(f)
    
    # Handle different model formats
    model = model_data.get('model')
    model_type = model_data.get('model_type', 'PerRetrieverLGBM')
    retrievers = retrievers or model_data.get('retrievers', sorted(runs.keys()))
    n_qpp = model_data.get('n_qpp', 13)
    
    # Build QPP features for all queries
    all_qids = sorted(set.union(*[set(df["qid"].unique()) for df in runs.values()]))
    n_retrievers = len(retrievers)
    
    # Build feature matrix
    X = np.zeros((len(all_qids), n_qpp * n_retrievers))
    for i, qid in enumerate(all_qids):
        for j, retriever in enumerate(retrievers):
            if qid in qpp_data and retriever in qpp_data[qid]:
                scores = qpp_data[qid][retriever]
                X[i, j*n_qpp:(j+1)*n_qpp] = scores[:n_qpp]
    
    # Predict weights using model's predict method
    pred_weights = model.predict(X)
    
    # Create weights dict
    weights_dict = {}
    for i, qid in enumerate(all_qids):
        weights_dict[qid] = {r: w for r, w in zip(retrievers, pred_weights[i])}
    
    # Fuse with learned weights - use vectorized approach
    all_dfs = []
    for ranker, df in runs.items():
        df_copy = df[["qid", "docno", "score"]].copy()
        # Apply per-query weights
        df_copy["weighted_score"] = df_copy.apply(
            lambda row: row["score"] * weights_dict.get(row["qid"], {}).get(ranker, 1.0/n_retrievers),
            axis=1
        )
        all_dfs.append(df_copy[["qid", "docno", "weighted_score"]])
    
    combined = pd.concat(all_dfs, ignore_index=True)
    aggregated = combined.groupby(["qid", "docno"])["weighted_score"].sum().reset_index()
    
    # Convert to dict format
    fused = defaultdict(list)
    for _, row in aggregated.iterrows():
        fused[row["qid"]].append((row["docno"], row["weighted_score"]))
    
    print(f"[fusion] Learned fusion done: {len(fused)} queries")
    return dict(fused)


# =============================================================================
# Utility Functions
# =============================================================================

def write_runfile(
    fused: Dict[str, List[Tuple[str, float]]],
    output_path: str,
    tag: str = "fusion"
):
    """Write fused results in TREC format."""
    with open(output_path, "w") as fout:
        for qid in sorted(fused.keys(), key=lambda x: int(x.replace("test", "")) if x.startswith("test") else x):
            ranked = sorted(fused[qid], key=lambda x: x[1], reverse=True)
            for rank, (docid, score) in enumerate(ranked, start=1):
                fout.write(f"{qid} Q0 {docid} {rank} {score:.6f} {tag}\n")
    
    print(f"Wrote fused run to {output_path}")


def run_fusion(
    method: str,
    runs_dir: str,
    qpp_dir: Optional[str] = None,
    qpp_model: str = "RSD",
    model_path: Optional[str] = None,
    output_path: Optional[str] = None,
    rrf_k: int = 60
) -> Dict[str, List[Tuple[str, float]]]:
    """
    Run specified fusion method.
    
    Args:
        method: One of combsum, combmnz, rrf, wcombsum, wcombmnz, wrrf, learned
        runs_dir: Directory with run files
        qpp_dir: Directory with QPP files (required for weighted methods)
        qpp_model: QPP model name for weighting
        model_path: Path to trained model (for learned fusion)
        output_path: Output file path (optional)
        rrf_k: RRF constant
    
    Returns:
        Fused results dict
    """
    runs = load_runs(runs_dir, use_normalized=True)
    print(f"Loaded {len(runs)} rankers: {list(runs.keys())}")
    
    method = method.lower()
    
    # Unweighted methods
    if method == "combsum":
        fused = combsum(runs)
        tag = "combsum"
        
    elif method == "combmnz":
        fused = combmnz(runs)
        tag = "combmnz"
        
    elif method == "rrf":
        fused = rrf(runs, k=rrf_k)
        tag = f"rrf-k{rrf_k}"
    
    # QPP-weighted methods
    elif method in ["wcombsum", "w-combsum"]:
        if not qpp_dir:
            raise ValueError("--qpp_dir required for weighted methods")
        qpp_data = load_qpp_scores(qpp_dir)
        qpp_index = get_qpp_index(qpp_model)
        fused = weighted_combsum(runs, qpp_data, qpp_index)
        tag = f"wcombsum-{qpp_model.lower()}"
        
    elif method in ["wcombmnz", "w-combmnz"]:
        if not qpp_dir:
            raise ValueError("--qpp_dir required for weighted methods")
        qpp_data = load_qpp_scores(qpp_dir)
        qpp_index = get_qpp_index(qpp_model)
        fused = weighted_combmnz(runs, qpp_data, qpp_index)
        tag = f"wcombmnz-{qpp_model.lower()}"
        
    elif method in ["wrrf", "w-rrf"]:
        if not qpp_dir:
            raise ValueError("--qpp_dir required for weighted methods")
        qpp_data = load_qpp_scores(qpp_dir)
        qpp_index = get_qpp_index(qpp_model)
        fused = weighted_rrf(runs, qpp_data, qpp_index, k=rrf_k)
        tag = f"wrrf-{qpp_model.lower()}"
    
    # Learned fusion
    elif method == "learned":
        if not model_path:
            raise ValueError("--model_path required for learned fusion")
        if not qpp_dir:
            raise ValueError("--qpp_dir required for learned fusion")
        qpp_data = load_qpp_scores(qpp_dir)
        fused = learned_fusion(runs, qpp_data, model_path)
        tag = "learned"
    
    else:
        valid = ["combsum", "combmnz", "rrf", "wcombsum", "wcombmnz", "wrrf", "learned"]
        raise ValueError(f"Unknown method '{method}'. Valid: {valid}")
    
    # Write output
    if output_path:
        write_runfile(fused, output_path, tag)
    
    return fused


# =============================================================================
# CLI
# =============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Fusion Methods for Multi-Retriever RAG",
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

Examples:
  python src/fusion.py --method combsum --runs_dir data/nq/runs --output fused.res
  python src/fusion.py --method wcombsum --runs_dir data/nq/runs --qpp_dir data/nq/qpp --qpp_model RSD
  python src/fusion.py --method learned --runs_dir data/nq/runs --qpp_dir data/nq/qpp --model_path models/fusion.pkl
"""
    )
    parser.add_argument("--method", required=True,
                        choices=["combsum", "combmnz", "rrf", "wcombsum", "wcombmnz", "wrrf", "learned"],
                        help="Fusion method")
    parser.add_argument("--runs_dir", required=True, help="Directory with .norm.res files")
    parser.add_argument("--qpp_dir", default=None, help="Directory with .qpp files")
    parser.add_argument("--qpp_model", default="RSD", help="QPP model for weighting")
    parser.add_argument("--model_path", default=None, help="Path to learned model")
    parser.add_argument("--output", required=True, help="Output TREC run file")
    parser.add_argument("--rrf_k", type=int, default=60, help="RRF k constant")
    
    args = parser.parse_args()
    
    run_fusion(
        method=args.method,
        runs_dir=args.runs_dir,
        qpp_dir=args.qpp_dir,
        qpp_model=args.qpp_model,
        model_path=args.model_path,
        output_path=args.output,
        rrf_k=args.rrf_k
    )


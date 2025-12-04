#!/usr/bin/env python3
"""
Incoming: documents with scores --- {List[Dict] with score field}
Processing: QPP computation via Java bridge --- {13 QPP methods}
Outgoing: QPP scores and predictions --- {Dict with qpp_scores}

Query Performance Prediction (QPP) Operations
----------------------------------------------
Implements 13 real QPP methods via Java bridge (QPPBridge.java):

1. NQC - Normalized Query Commitment
2. SMV - Similarity Mean Variance  
3. WIG - Weighted Information Gain
4. SigmaMax - Maximum Standard Deviation
5. SigmaX - Threshold-based Std Dev
6. RSD - Retrieval Score Distribution
7. UEF - Utility Estimation Framework
8. MaxIDF - Maximum IDF
9. AvgIDF - Average IDF
10. CumNQC - Cumulative NQC
11. SNQC - Calibrated NQC
12. DenseQPP - Dense Vector QPP
13. DenseQPP-M - Matryoshka Dense QPP

Usage:
    from src.qpp import QPPBridge
    qpp = QPPBridge()
    result = qpp.compute(query="what is X", scores=[0.9, 0.7, 0.5, ...])
"""

import os
import sys
import json
import subprocess
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass


# QPP method index mapping (matches Combsum.py and Java QPPBridge)
QPP_METHODS = {
    0: "SMV", 1: "Sigma_max", 2: "Sigma(%)", 3: "NQC", 4: "UEF", 5: "RSD",
    6: "QPP-PRP", 7: "WIG", 8: "SCNQC", 9: "QV-NQC", 10: "DM",
    11: "NQA-QPP", 12: "BERTQPP"
}

QPP_METHOD_NAMES = [
    'nqc', 'smv', 'wig', 'SigmaMax', 'SigmaX', 'RSD', 'UEF',
    'MaxIDF', 'avgidf', 'cumnqc', 'snqc', 'dense-qpp', 'dense-qpp-m'
]


@dataclass
class QPPResult:
    """QPP computation result."""
    query: str
    retriever_name: str
    qpp_scores: Dict[str, float]
    methods_used: List[str]
    processing_time_ms: float
    predictions: Dict[str, Any]
    error: Optional[str] = None


class QPPBridge:
    """
    Python-Java bridge for QPP computation.
    
    Automatically discovers and uses Java QPPBridge.class if available,
    falls back to pure Python implementation otherwise.
    """
    
    def __init__(self, java_dir: Optional[str] = None):
        """
        Initialize QPP Bridge.
        
        Args:
            java_dir: Optional path to Java QPP directory. 
                      Defaults to src/qpp relative to this file.
        """
        self.src_dir = Path(__file__).parent
        self.java_dir = Path(java_dir) if java_dir else self.src_dir / "qpp"
        self.lib_dir = self.src_dir.parent / "lib"
        
        # Check for compiled Java
        self.java_available = self._check_java()
        
        if self.java_available:
            print(f"✅ Java QPP bridge available at {self.java_dir}", file=sys.stderr)
        else:
            print(f"⚠️  Java QPP not compiled. Using Python fallback.", file=sys.stderr)
            print(f"   To compile: cd {self.java_dir} && ./build.sh", file=sys.stderr)
    
    def _check_java(self) -> bool:
        """Check if Java QPPBridge is compiled and available."""
        # Check for compiled class file
        class_file = self.java_dir / "QPPBridge.class"
        if class_file.exists():
            return True
        
        # Check for JAR in lib/
        jar_file = self.lib_dir / "qpp-bridge.jar"
        if jar_file.exists():
            return True
        
        return False
    
    def _get_classpath(self) -> str:
        """Build Java classpath."""
        paths = []
        
        # Add compiled classes directory (parent of qpp/ for package structure)
        if (self.java_dir / "QPPBridge.class").exists():
            paths.append(str(self.java_dir.parent))
        
        # Add JAR if exists
        jar_file = self.lib_dir / "qpp-bridge.jar"
        if jar_file.exists():
            paths.append(str(jar_file))
        
        # Add Gson dependency - check multiple versions
        for gson_name in ["gson-2.11.0.jar", "gson-2.10.1.jar", "gson.jar"]:
            gson_jar = self.lib_dir / gson_name
            if gson_jar.exists():
                paths.append(str(gson_jar))
                break
        
        return ":".join(paths)
    
    def compute(
        self,
        query: str,
        scores: List[float],
        retriever_name: str = "unknown",
        methods: Optional[List[str]] = None,
        use_java: bool = True
    ) -> QPPResult:
        """
        Compute QPP scores for a query's retrieval scores.
        
        Args:
            query: Query text
            scores: List of retrieval scores (top-k documents)
            retriever_name: Name of retriever
            methods: List of QPP methods to compute (default: all 13)
            use_java: Whether to try Java bridge first
            
        Returns:
            QPPResult with all QPP scores
        """
        methods = methods or QPP_METHOD_NAMES
        
        if use_java and self.java_available:
            try:
                return self._compute_java(query, scores, retriever_name, methods)
            except Exception as e:
                print(f"⚠️  Java QPP failed: {e}, using Python", file=sys.stderr)
        
        return self._compute_python(query, scores, retriever_name, methods)
    
    def _compute_java(
        self,
        query: str,
        scores: List[float],
        retriever_name: str,
        methods: List[str]
    ) -> QPPResult:
        """Compute QPP via Java subprocess."""
        import time
        start = time.time()
        
        # Build input JSON
        input_data = {
            "query": query,
            "documents": [{"score": s} for s in scores],
            "retriever_name": retriever_name,
            "methods": methods
        }
        
        classpath = self._get_classpath()
        
        result = subprocess.run(
            ["java", "-cp", classpath, "qpp.QPPBridge"],
            input=json.dumps(input_data),
            capture_output=True,
            text=True,
            timeout=30
        )
        
        if result.returncode != 0:
            raise RuntimeError(f"Java QPP failed: {result.stderr}")
        
        output = json.loads(result.stdout)
        
        return QPPResult(
            query=query,
            retriever_name=retriever_name,
            qpp_scores=output.get("qpp_scores", {}),
            methods_used=output.get("methods_used", []),
            processing_time_ms=(time.time() - start) * 1000,
            predictions=output.get("predictions", {})
        )
    
    def _compute_python(
        self,
        query: str,
        scores: List[float],
        retriever_name: str,
        methods: List[str]
    ) -> QPPResult:
        """Pure Python QPP implementation (fallback)."""
        import time
        start = time.time()
        
        qpp_scores = {}
        query_len = len(query.split())
        
        for method in methods:
            try:
                score = self._compute_single_method(method, scores, query_len)
                qpp_scores[method] = score
            except Exception as e:
                print(f"Warning: {method} failed: {e}", file=sys.stderr)
                qpp_scores[method] = 0.0
        
        # Aggregate predictions
        predictions = self._aggregate_predictions(qpp_scores)
        
        return QPPResult(
            query=query,
            retriever_name=retriever_name,
            qpp_scores=qpp_scores,
            methods_used=list(qpp_scores.keys()),
            processing_time_ms=(time.time() - start) * 1000,
            predictions=predictions
        )
    
    def _compute_single_method(self, method: str, scores: List[float], query_len: int) -> float:
        """Compute single QPP method in Python."""
        if len(scores) < 2:
            return 0.5
        
        arr = np.array(scores)
        
        if method == "nqc":
            return self._nqc(arr, k=50)
        elif method == "smv":
            return self._smv(arr, k=10)
        elif method == "wig":
            return self._wig(arr, query_len, k=20)
        elif method == "SigmaMax":
            return self._sigma_max(arr, query_len, k=50)
        elif method == "SigmaX":
            return self._sigma_x(arr, k=50)
        elif method == "RSD":
            return self._rsd(arr)
        elif method == "UEF":
            return self._uef(arr, k=20)
        elif method == "MaxIDF":
            return min(1.0, np.log(1 + query_len) / 5.0)
        elif method == "avgidf":
            return min(1.0, np.log(1 + query_len) * 0.8 / 5.0)
        elif method == "cumnqc":
            return self._cumnqc(arr, k=50)
        elif method == "snqc":
            return self._snqc(arr, k=50)
        elif method == "dense-qpp":
            return self._dense_qpp(arr, k=5)
        elif method == "dense-qpp-m":
            return self._dense_qpp_m(arr, k=5)
        else:
            raise ValueError(f"Unknown QPP method: {method}")
    
    # ========================================================================
    # Python QPP Method Implementations (mirrors Java)
    # ========================================================================
    
    def _nqc(self, scores: np.ndarray, k: int = 50) -> float:
        """NQC: Normalized Query Commitment."""
        top_k = scores[:min(k, len(scores))]
        variance = np.var(top_k)
        return 1.0 / (1.0 + np.exp(-variance))
    
    def _smv(self, scores: np.ndarray, k: int = 10) -> float:
        """SMV: Similarity Mean Variance."""
        top_k = scores[:min(k, len(scores))]
        mu = np.mean(top_k)
        if mu < 1e-10:
            return 0.0
        smv = np.mean(top_k * np.abs(np.log(top_k / mu + 1e-10)))
        return 1.0 / (1.0 + np.exp(-smv))
    
    def _wig(self, scores: np.ndarray, query_len: int, k: int = 20) -> float:
        """WIG: Weighted Information Gain."""
        top_k = scores[:min(k, len(scores))]
        avg_idf = 1.0
        wig = np.sum(top_k - avg_idf) / (query_len * len(top_k))
        return 1.0 / (1.0 + np.exp(-wig * 10))
    
    def _sigma_max(self, scores: np.ndarray, query_len: int, k: int = 50) -> float:
        """SigmaMax: Maximum standard deviation."""
        top_k = scores[:min(k, len(scores))]
        max_std = max(np.std(top_k[:i]) for i in range(2, len(top_k) + 1))
        norm = np.sqrt(max(1, query_len))
        return 1.0 / (1.0 + np.exp(-max_std / norm * 2))
    
    def _sigma_x(self, scores: np.ndarray, k: int = 50) -> float:
        """SigmaX: Threshold-based std dev."""
        top_k = scores[:min(k, len(scores))]
        threshold = top_k[0] * 0.5
        filtered = top_k[top_k >= threshold]
        if len(filtered) == 0:
            return 0.0
        std = np.std(filtered)
        return 1.0 / (1.0 + np.exp(-std * 2))
    
    def _rsd(self, scores: np.ndarray) -> float:
        """RSD: Retrieval Score Distribution (skewness)."""
        mean = np.mean(scores)
        std = np.std(scores)
        if std < 1e-10:
            return min(1.0, mean)
        skewness = np.mean(((scores - mean) / std) ** 3)
        return 1.0 / (1.0 + np.exp(-skewness))
    
    def _uef(self, scores: np.ndarray, k: int = 20) -> float:
        """UEF: Utility Estimation Framework."""
        top_k = scores[:min(k, len(scores))]
        weights = 1.0 / np.arange(1, len(top_k) + 1)
        utility = np.sum(top_k * weights) / np.sum(weights)
        return min(1.0, utility)
    
    def _cumnqc(self, scores: np.ndarray, k: int = 50) -> float:
        """CumNQC: Cumulative NQC."""
        top_k = scores[:min(k, len(scores))]
        cum_sum = sum(np.var(top_k[:i]) for i in range(2, len(top_k) + 1))
        return 1.0 / (1.0 + np.exp(-cum_sum / k * 10))
    
    def _snqc(self, scores: np.ndarray, k: int = 50) -> float:
        """SNQC: Calibrated NQC."""
        top_k = scores[:min(k, len(scores))]
        mean = np.mean(top_k)
        alpha, beta, gamma = 0.33, 0.33, 0.33
        snqc = 0.0
        for rsv in top_k:
            if rsv > 0:
                f1 = 1.0 ** alpha  # avgIDF = 1
                f2 = ((rsv - mean) ** 2 / rsv) ** beta
                snqc += (f1 * f2) ** gamma
        snqc = snqc / len(top_k) * 1.0  # * avgIDF
        return 1.0 / (1.0 + np.exp(-snqc * 10))
    
    def _dense_qpp(self, scores: np.ndarray, k: int = 5) -> float:
        """DenseQPP: Dense vector QPP."""
        top_k = scores[:min(k, len(scores))]
        variance = np.var(top_k)
        diameter = np.sqrt(variance) + 0.01
        dense_qpp = np.log(1 + 1 / diameter)
        return min(1.0, dense_qpp / 5.0)
    
    def _dense_qpp_m(self, scores: np.ndarray, k: int = 5) -> float:
        """DenseQPP-M: Matryoshka dense QPP."""
        top_k = scores[:min(k, len(scores))]
        weighted_sum = 0.0
        for i in range(1, len(top_k) + 1):
            window = top_k[:i]
            variance = np.var(window)
            diameter = np.sqrt(variance) + 0.01
            weight = 1.0 / np.log(1 + i)
            weighted_sum += weight * np.log(1 + 1 / diameter)
        return min(1.0, weighted_sum / k)
    
    def _aggregate_predictions(self, qpp_scores: Dict[str, float]) -> Dict[str, Any]:
        """Aggregate QPP scores into predictions."""
        if not qpp_scores:
            return {
                "difficulty_estimate": 1.0,
                "retrieval_quality": 0.0,
                "recommended_action": "fallback",
                "confidence": 0.0
            }
        
        scores = list(qpp_scores.values())
        mean_qpp = np.mean(scores)
        variance = np.var(scores)
        
        confidence = 1.0 / (1.0 + variance)
        retrieval_quality = mean_qpp
        difficulty_estimate = 1.0 - retrieval_quality
        
        if retrieval_quality >= 0.7:
            action = "proceed"
        elif retrieval_quality >= 0.4:
            action = "augment"
        else:
            action = "fallback"
        
        return {
            "difficulty_estimate": difficulty_estimate,
            "retrieval_quality": retrieval_quality,
            "recommended_action": action,
            "confidence": confidence
        }


# ============================================================================
# Batch QPP Computation (for .res files)
# ============================================================================

def compute_qpp_for_res_file(
    res_path: str,
    output_path: Optional[str] = None,
    top_k: int = 100,
    normalize: str = "minmax"
) -> Dict[str, List[float]]:
    """
    Compute QPP scores for all queries in a TREC .res file.
    
    Args:
        res_path: Path to .res file
        output_path: Path for .qpp output (optional)
        top_k: Top-k documents for QPP
        normalize: "minmax", "zscore", or "none"
        
    Returns:
        Dict of {qid: [13 QPP scores]}
    """
    from collections import defaultdict
    
    # Load run file
    runs = defaultdict(list)
    with open(res_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 5:
                qid, _, docno, rank, score = parts[:5]
                runs[qid].append(float(score))
    
    # Sort and truncate
    for qid in runs:
        runs[qid] = sorted(runs[qid], reverse=True)[:top_k]
    
    # Compute QPP
    bridge = QPPBridge()
    results = {}
    
    for qid, scores in runs.items():
        result = bridge.compute(query=qid, scores=scores)  # Use qid as placeholder query
        # Convert to list in QPP_METHODS order
        qpp_list = [result.qpp_scores.get(m, 0.0) for m in QPP_METHOD_NAMES]
        results[qid] = qpp_list
    
    # Normalize
    if normalize != "none" and results:
        results = _normalize_qpp(results, normalize)
    
    # Write output
    if output_path:
        with open(output_path, 'w') as f:
            for qid in sorted(results.keys(), key=lambda x: int(x) if x.isdigit() else x):
                scores = results[qid]
                score_str = '\t'.join(f"{s:.6f}" for s in scores)
                f.write(f"{qid}\t{score_str}\n")
        print(f"Wrote QPP scores to {output_path}")
    
    return results


def _normalize_qpp(results: Dict[str, List[float]], method: str) -> Dict[str, List[float]]:
    """Normalize QPP scores across queries."""
    n_methods = len(QPP_METHOD_NAMES)
    
    # Collect per-method values
    method_values = [[] for _ in range(n_methods)]
    for scores in results.values():
        for i, score in enumerate(scores):
            method_values[i].append(score)
    
    # Compute params
    params = []
    for values in method_values:
        arr = np.array(values)
        if method == "minmax":
            vmin, vmax = arr.min(), arr.max()
            params.append((vmin, vmax - vmin if vmax > vmin else 1.0))
        else:
            params.append((arr.mean(), arr.std() if arr.std() > 0 else 1.0))
    
    # Apply normalization
    normalized = {}
    for qid, scores in results.items():
        norm_scores = []
        for i, score in enumerate(scores):
            vmin, scale = params[i]
            if method == "minmax":
                norm_scores.append((score - vmin) / scale if scale > 0 else 0.0)
            else:
                norm_scores.append((score - vmin) / scale)
        normalized[qid] = norm_scores
    
    return normalized


# ============================================================================
# CLI
# ============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="QPP Bridge - Compute 13 QPP methods")
    parser.add_argument("--res_file", help="TREC .res file to process")
    parser.add_argument("--output", help="Output .qpp file")
    parser.add_argument("--top_k", type=int, default=100, help="Top-k for QPP")
    parser.add_argument("--normalize", choices=["none", "minmax", "zscore"], 
                        default="minmax", help="Normalization")
    parser.add_argument("--test", action="store_true", help="Run test")
    args = parser.parse_args()
    
    if args.test:
        # Quick test
        bridge = QPPBridge()
        result = bridge.compute(
            query="what is machine learning",
            scores=[0.95, 0.82, 0.71, 0.65, 0.58, 0.52, 0.48, 0.41, 0.35, 0.28]
        )
        print(f"Query: {result.query}")
        print(f"QPP Scores:")
        for method, score in result.qpp_scores.items():
            print(f"  {method}: {score:.4f}")
        print(f"Predictions: {result.predictions}")
        
    elif args.res_file:
        output = args.output or args.res_file.replace(".res", ".mmnorm.qpp")
        compute_qpp_for_res_file(args.res_file, output, args.top_k, args.normalize)
    else:
        parser.print_help()

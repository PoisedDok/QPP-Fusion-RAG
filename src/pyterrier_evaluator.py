"""
PyTerrier Evaluator Operation

Evaluates retrieval results against ground-truth qrels using standard IR metrics.
"""

from __future__ import annotations

import time
from typing import Dict, Any, List, Optional

try:
    import pyterrier as pt  # type: ignore
    import pandas as pd

    if not pt.started():
        pt.init()

    PYT_AVAILABLE = True
except Exception as e:
    PYT_AVAILABLE = False
    _IMPORT_ERROR = str(e)


class PyTerrierEvaluatorOperation:
    """Evaluate retrieval results using PyTerrier / ir-measures."""

    def __init__(self, executor=None):
        if not PYT_AVAILABLE:
            raise RuntimeError("pyterrier not available: " + _IMPORT_ERROR)

    def execute(
        self,
        results: List[Dict[str, Any]],
        qrels: List[Dict[str, Any]],
        metrics: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Evaluate retrieval results.

        Args:
            results: List of {qid, docno, score, rank} dicts.
            qrels: List of {qid, docno, label} dicts.
            metrics: List of metric names (e.g., ['map', 'ndcg@10', 'mrr']).

        Returns:
            Dict with per-query and overall metrics.
        """
        start = time.time()

        if not results or not qrels:
            return {"error": "Empty results or qrels", "metrics": {}}

        # Default metrics
        if not metrics:
            metrics = ["map", "ndcg@10", "ndcg@20", "recip_rank", "P@10"]

        # Convert to DataFrames
        results_df = pd.DataFrame(results)
        qrels_df = pd.DataFrame(qrels)

        # Ensure correct columns
        if "qid" not in results_df.columns or "docno" not in results_df.columns:
            return {"error": "Results must have qid and docno columns", "metrics": {}}
        if "qid" not in qrels_df.columns or "docno" not in qrels_df.columns or "label" not in qrels_df.columns:
            return {"error": "Qrels must have qid, docno, label columns", "metrics": {}}

        try:
            # Use PyTerrier's evaluation
            eval_results = pt.Experiment(
                [results_df],
                topics=qrels_df[["qid", "query"]].drop_duplicates() if "query" in qrels_df.columns else None,
                qrels=qrels_df,
                eval_metrics=metrics,
                names=["system"],
            )
            
            # Convert to dict
            metrics_dict = eval_results.to_dict(orient="records")[0] if len(eval_results) > 0 else {}
        except Exception as e:
            # Fallback: use ir_measures directly
            try:
                import ir_measures  # type: ignore
                from ir_measures import calc_aggregate  # type: ignore

                # Convert metric names
                ir_metrics = [ir_measures.parse_measure(m) for m in metrics]
                
                # Build run and qrels in ir_measures format
                run_list = [
                    (row["qid"], row["docno"], row.get("score", 0.0))
                    for _, row in results_df.iterrows()
                ]
                qrels_list = [
                    (row["qid"], row["docno"], int(row["label"]))
                    for _, row in qrels_df.iterrows()
                ]

                aggregated = calc_aggregate(ir_metrics, qrels_list, run_list)
                metrics_dict = {str(k): float(v) for k, v in aggregated.items()}
            except Exception as fallback_e:
                return {"error": f"Evaluation failed: {str(e)}, fallback: {str(fallback_e)}", "metrics": {}}

        elapsed = time.time() - start

        return {
            "metrics": metrics_dict,
            "query_count": len(results_df["qid"].unique()) if "qid" in results_df.columns else 0,
            "processing_time_ms": elapsed * 1000,
        }


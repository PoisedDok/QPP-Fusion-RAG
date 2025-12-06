"""
Incoming: runs, qrels, predictions, references --- {TREC format, gold labels}
Processing: evaluation --- {IR metrics, QA metrics}
Outgoing: metric scores --- {Dict[str, float]}

Evaluation Module
-----------------
Unified evaluation using research-grade packages:
- IR metrics: ir_measures (NDCG, MRR, Recall, MAP, etc.)
- QA metrics: HuggingFace evaluate (EM, F1, ROUGE)
"""

from .ir_evaluator import IREvaluator, evaluate_run, evaluate_runs
from .qa_evaluator import QAEvaluator, compute_qa_metrics

__all__ = [
    "IREvaluator",
    "evaluate_run",
    "evaluate_runs",
    "QAEvaluator", 
    "compute_qa_metrics",
]


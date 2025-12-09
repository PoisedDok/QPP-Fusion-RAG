"""
Incoming: runs, qrels, predictions, references --- {TREC format, gold labels}
Processing: evaluation --- {IR metrics, QA metrics, fact verification metrics}
Outgoing: metric scores --- {Dict[str, float]}

Evaluation Module
-----------------
Unified evaluation using research-grade packages:
- IR metrics: ir_measures (NDCG, MRR, Recall, MAP, etc.)
- QA metrics: HuggingFace evaluate (EM, F1, ROUGE)
- Fact Verification: 3-way classification (SUPPORT, CONTRADICT, NOT_ENOUGH_INFO)

Task Types:
- QA: NQ, HotpotQA, TriviaQA (answer extraction/generation)
- FactVerification: SciFact, FEVER (claim verification)
"""

from .base import (
    TaskType,
    GoldLabel,
    QAGoldLabel,
    FactVerificationGoldLabel,
    Prediction,
    QAPrediction,
    FactVerificationPrediction,
    TaskEvaluator,
    get_task_type,
)
from .ir_evaluator import IREvaluator, evaluate_run, evaluate_runs
from .qa_evaluator import QAEvaluator, compute_qa_metrics
from .fact_verification import (
    FactVerificationEvaluator,
    compute_fact_verification_metrics,
)

__all__ = [
    # Base classes
    "TaskType",
    "GoldLabel",
    "QAGoldLabel",
    "FactVerificationGoldLabel",
    "Prediction",
    "QAPrediction",
    "FactVerificationPrediction",
    "TaskEvaluator",
    "get_task_type",
    # IR evaluation
    "IREvaluator",
    "evaluate_run",
    "evaluate_runs",
    # QA evaluation
    "QAEvaluator", 
    "compute_qa_metrics",
    # Fact verification
    "FactVerificationEvaluator",
    "compute_fact_verification_metrics",
]


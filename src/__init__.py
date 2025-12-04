"""
Incoming: none --- {none}
Processing: package initialization --- {1 job: exports}
Outgoing: submodules --- {Python modules}

QPP-Fusion-RAG: Query Performance Prediction Guided Retrieval Fusion

Clean standalone implementation for ECIR paper reproduction.
"""

__version__ = "1.1.0"

# QPP Bridge (13 methods via Java + Python fallback)
from .qpp import QPPBridge, QPP_METHOD_NAMES, compute_qpp_for_res_file

# LM Studio Generation
from .generation import GenerationOperation

# K-Shot RAG Experiment
from .kshot_rag import KShotRAGExperimentOperation

# Dataset Loading
from .datasets import DatasetLoadOperation

# QA Evaluation
from .qa_evaluation import QAEvaluationOperation

# TREC Format
from .trec_run_writer import TRECRunWriterOperation

__all__ = [
    # QPP
    "QPPBridge",
    "QPP_METHOD_NAMES",
    "compute_qpp_for_res_file",
    # Generation
    "GenerationOperation",
    # RAG
    "KShotRAGExperimentOperation",
    # Data
    "DatasetLoadOperation",
    # Evaluation
    "QAEvaluationOperation",
    # TREC
    "TRECRunWriterOperation",
]

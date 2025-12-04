"""
TREC Run File Writer

Converts retrieval results to TREC run format (.res files).
"""

from __future__ import annotations

import os
import time
from typing import Dict, Any, List, Optional


class TRECRunWriterOperation:
    """Write retrieval results in TREC run format."""

    def __init__(self, executor=None):
        pass

    def execute(
        self,
        results: List[Dict[str, Any]],
        output_path: str,
        run_name: str = "system",
        qid: Optional[str] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Write results to TREC run file.

        Args:
            results: List of results with qid, docno, score, rank fields.
            output_path: Path to output .res file.
            run_name: Run name (system identifier).
            qid: Optional single query ID (if all results are for one query).

        Returns:
            Dict with status and file path.
        """
        start = time.time()

        if not results:
            return {"status": "failed", "error": "No results to write", "output_path": None}

        try:
            # Ensure output directory exists
            os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

            with open(output_path, "w", encoding="utf-8") as f:
                for i, result in enumerate(results):
                    result_qid = result.get("qid", qid or "Q1")
                    docno = result.get("docno", f"doc{i}")
                    score = result.get("score", 0.0)
                    rank = result.get("rank", i + 1)

                    # TREC format: qid Q0 docno rank score run_name
                    f.write(f"{result_qid} Q0 {docno} {rank} {score:.6f} {run_name}\n")

            elapsed = time.time() - start

            return {
                "status": "success",
                "output_path": output_path,
                "lines_written": len(results),
                "run_name": run_name,
                "processing_time_ms": elapsed * 1000,
            }

        except Exception as e:
            return {
                "status": "failed",
                "error": str(e),
                "output_path": None,
            }


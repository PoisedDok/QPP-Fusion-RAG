"""
PyTerrier Batch Indexer

Creates multiple indexes with different encoders in parallel for multi-retriever experiments.
"""

from __future__ import annotations

import os
import time
import tempfile
from typing import Dict, Any, List, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

try:
    import pyterrier as pt  # type: ignore
    import pyterrier_dr as dr  # type: ignore

    if not pt.started():
        pt.init()

    PYT_AVAILABLE = True
except Exception as e:
    PYT_AVAILABLE = False
    _IMPORT_ERROR = str(e)


class PyTerrierBatchIndexOperation:
    """Index documents with multiple encoders simultaneously."""

    def __init__(self, executor=None):
        if not PYT_AVAILABLE:
            raise RuntimeError("pyterrier or pyterrier_dr not available: " + _IMPORT_ERROR)

    def execute(
        self,
        documents: List[Dict[str, Any]],
        encoders: List[str],
        base_path: Optional[str] = None,
        batch_size: int = 32,
        max_workers: int = 4,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Index documents with multiple encoders.

        Args:
            documents: List of documents.
            encoders: List of encoder model names.
            base_path: Base directory for indexes.
            batch_size: Batch size for encoding.
            max_workers: Number of parallel indexing jobs.

        Returns:
            Dict with index_paths mapping encoder -> path.
        """
        start = time.time()

        if not documents:
            return {"status": "failed", "error": "No documents", "index_paths": {}}

        if not encoders:
            return {"status": "failed", "error": "No encoders specified", "index_paths": {}}

        # Prepare DataFrame
        import pandas as pd
        df = pd.DataFrame(
            [[i, doc.get("content", "")] for i, doc in enumerate(documents)],
            columns=["docno", "text"],
        )

        base_dir = base_path or tempfile.mkdtemp(prefix="ptdr_batch_")
        os.makedirs(base_dir, exist_ok=True)

        index_paths = {}
        errors = {}

        def index_with_encoder(encoder_name: str) -> tuple[str, str]:
            """Index with single encoder."""
            try:
                # Sanitize encoder name for filesystem
                safe_name = encoder_name.replace("/", "_").replace(":", "_")
                index_dir = os.path.join(base_dir, safe_name)
                os.makedirs(index_dir, exist_ok=True)

                indexer = dr.DPRIndexer(encoder_name, batch_size=batch_size, index_path=index_dir)
                indexer.index(df)

                return (encoder_name, index_dir)
            except Exception as e:
                return (encoder_name, f"ERROR: {str(e)}")

        # Index in parallel
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(index_with_encoder, enc): enc for enc in encoders}

            for future in as_completed(futures):
                encoder_name, result = future.result()
                if result.startswith("ERROR:"):
                    errors[encoder_name] = result
                else:
                    index_paths[encoder_name] = result

        elapsed = time.time() - start

        return {
            "status": "success" if index_paths else "failed",
            "indexed_count": len(df),
            "index_paths": index_paths,
            "errors": errors if errors else None,
            "encoders_completed": len(index_paths),
            "encoders_failed": len(errors),
            "processing_time_ms": elapsed * 1000,
        }


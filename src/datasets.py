"""
Dataset loading operations for standard IR/RAG benchmarks.

Supports:
- FEVER
- SciFact
- NaturalQuestions (NQ)
- TriviaQA
- TREC Deep Learning 2019/2020 (TREC DL)

Two loading modes:
1) Built-in via ir_datasets/python-terrier (preferred)
2) Local files provided by the user (qrels/queries/corpus)

Outputs a normalized JSON with queries, qrels, and documents (corpus) for downstream blocks.
"""

import json
import time
from typing import Any, Dict, List, Optional, Tuple


class DatasetLoadOperation:
    """Load standard IR datasets into a normalized JSON format."""

    def __init__(self, executor=None):
        self.executor = executor

        # Lazy flags for optional deps
        self._ir_datasets = None
        self._pt = None

    def execute(
        self,
        dataset: str = "trec_dl_2019",
        split: str = "test",
        limit: Optional[int] = None,
        source_files: Optional[Dict[str, str]] = None,
        folder_files: Optional[List[str]] = None,
        include_corpus: bool = False,
        data_home: Optional[str] = None,
        ensure_download: bool = True,
        preload_all: bool = False,
        folder_path: Optional[str] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        start_time = time.time()

        try:
            import sys
            print(f"[DatasetLoader] Detecting dataset '{dataset}' split '{split}'", flush=True, file=sys.stderr)

            # Optional: set data home paths for ir_datasets / PyTerrier
            if data_home:
                import os
                os.environ["IR_DATASETS_HOME"] = data_home
                os.environ["PYTERRIER_DATA"] = data_home
                os.environ["PYTERRIER_HOME"] = data_home

            # Determine dataset format and file locations
            if folder_path:
                result = self._detect_local_dataset(folder_path, split)
            elif folder_files and len(folder_files) > 0:
                sf = self._autodetect_files(folder_files)
                result = self._detect_from_source_files(sf, split)
            elif source_files and len(source_files) > 0:
                result = self._detect_from_source_files(source_files, split)
            else:
                if dataset and (dataset.startswith('/') or dataset.startswith('~')):
                    result = self._detect_local_dataset(dataset, split)
                else:
                    result = self._detect_ir_datasets(dataset, split, ensure_download)

            result["processing_time_ms"] = (time.time() - start_time) * 1000
            import sys
            print(f"[DatasetLoader] Detection completed in {result['processing_time_ms']:.1f}ms. "
                  f"Format: {result['format']}, Counts: {result['counts']}", flush=True, file=sys.stderr)

            return result

        except Exception as e:
            import sys
            print(f"[DatasetLoader] Error detecting dataset: {e}", flush=True, file=sys.stderr)
            return {
                "error": str(e),
                "error_type": type(e).__name__,
                "dataset": dataset,
                "split": split,
            }

    # ---------------------------------------------------------------------
    # Helpers
    # ---------------------------------------------------------------------

    def _lazy_imports(self):
        if self._ir_datasets is None:
            try:
                import ir_datasets  # type: ignore

                self._ir_datasets = ir_datasets
            except Exception:
                self._ir_datasets = None

        if self._pt is None:
            try:
                import pyterrier as pt  # type: ignore

                if not pt.started():
                    pt.init()
                self._pt = pt
            except Exception:
                self._pt = None

    def _normalize_dataset_name(self, dataset: str) -> str:
        d = dataset.lower().strip()

        # If it's already a full path, return as-is
        if '/' in d:
            return d

        aliases = {
            "trec_dl_2019": "msmarco-passage/trec-dl-2019/judged",
            "trecdl2019": "msmarco-passage/trec-dl-2019/judged",
            "trecdl19": "msmarco-passage/trec-dl-2019/judged",
            "trec_dl_2020": "msmarco-passage/trec-dl-2020/judged",
            "trecdl2020": "msmarco-passage/trec-dl-2020/judged",
            "trecdl20": "msmarco-passage/trec-dl-2020/judged",
            "scifact": "beir/scifact",
            "fever": "beir/fever",
            "nq": "beir/natural-questions",
            "naturalquestions": "beir/natural-questions",
            "natural-questions": "beir/natural-questions",
            "triviaqa": "beir/triviaqa",
            "fiqa": "beir/fiqa",
            "quora": "beir/quora",
            "dbpedia": "beir/dbpedia-entity",
            "msmarco-passage-trec-dl-2019": "msmarco-passage/trec-dl-2019/judged",
            "msmarco-passage-trec-dl-2020": "msmarco-passage/trec-dl-2020/judged",
        }
        return aliases.get(d, f"beir/{d}")

    def _detect_ir_datasets(self, dataset: str, split: str, ensure_download: bool) -> Dict[str, Any]:
        """Detect ir_datasets availability and return metadata without loading data."""
        self._lazy_imports()
        if self._ir_datasets is None:
            raise RuntimeError("ir_datasets not installed")

        ir_datasets = self._ir_datasets
        ds_name = self._normalize_dataset_name(dataset)

        try:
            ds = ir_datasets.load(ds_name)
        except Exception as e:
            raise RuntimeError(f"Failed to load dataset '{ds_name}' via ir_datasets: {e}")

        # Count items efficiently without loading full data
        counts = self._count_from_ir_dataset(ds, split)

        return {
            "dataset": dataset,
            "split": split,
            "format": "ir_datasets",
            "paths": {
                "dataset_name": ds_name,
                "ir_datasets_id": ds_name,
            },
            "counts": counts,
        }

    def _detect_local_dataset(self, folder_path: str, split: str) -> Dict[str, Any]:
        """Detect local dataset structure and return metadata."""
        import os

        sf = self._autodetect_from_folder(folder_path)
        counts = self._count_from_files(sf, split)

        return {
            "dataset": os.path.basename(folder_path),
            "split": split,
            "format": "local",
            "paths": {
                "base_path": folder_path,
                **sf
            },
            "counts": counts,
        }

    def _detect_from_source_files(self, source_files: Dict[str, str], split: str) -> Dict[str, Any]:
        """Detect from explicit source files and return metadata."""
        counts = self._count_from_files(source_files, split)

        return {
            "dataset": "custom",
            "split": split,
            "format": "local",
            "paths": source_files,
            "counts": counts,
        }

    def _count_from_ir_dataset(self, ds, split: str) -> Dict[str, int]:
        """Count items in ir_dataset efficiently without loading full content."""
        counts = {"queries": 0, "qrels": 0, "documents": 0}

        try:
            queries_iter = getattr(ds, "queries_iter", None)
            if queries_iter:
                counts["queries"] = sum(1 for _ in queries_iter())
        except:
            pass

        try:
            qrels_iter = getattr(ds, "qrels_iter", None)
            if qrels_iter:
                counts["qrels"] = sum(1 for _ in qrels_iter())
        except:
            pass

        try:
            if hasattr(ds, "docs_iter"):
                counts["documents"] = sum(1 for _ in ds.docs_iter())
        except:
            pass

        return counts

    def _count_from_files(self, files: Dict[str, str], split: str) -> Dict[str, int]:
        """Count items in files efficiently without loading full content."""
        import os
        counts = {"queries": 0, "qrels": 0, "documents": 0}

        # Count queries
        qp = files.get("queries_path")
        if qp and os.path.exists(qp):
            _, ext = os.path.splitext(qp)
            ext = ext.lower()
            with open(qp, "r", encoding="utf-8") as f:
                if ext == ".jsonl":
                    counts["queries"] = sum(1 for line in f if line.strip())
                else:
                    counts["queries"] = sum(1 for line in f if line.strip())

        # Count qrels
        rp = files.get("qrels_path")
        if rp and os.path.exists(rp):
            with open(rp, "r", encoding="utf-8") as f:
                counts["qrels"] = sum(1 for line in f if line.strip() and not line.startswith("#"))

        # Count documents
        cp = files.get("corpus_path")
        if cp and os.path.exists(cp):
            _, ext = os.path.splitext(cp)
            ext = ext.lower()
            with open(cp, "r", encoding="utf-8") as f:
                if ext == ".jsonl":
                    counts["documents"] = sum(1 for line in f if line.strip())
                else:
                    counts["documents"] = sum(1 for line in f if line.strip())

        return counts

    def _preload_standard_sets(self, data_home: Optional[str]):
        """Download a selected list of standard datasets via ir_datasets/PyTerrier."""
        self._lazy_imports()
        try:
            import os
            if data_home:
                os.environ["IR_DATASETS_HOME"] = data_home
                os.environ["PYTERRIER_DATA"] = data_home
                os.environ["PYTERRIER_HOME"] = data_home

            # Touch datasets so they are cached
            standard = [
                "fever",
                "scifact",
                "natural-questions",
                "triviaqa",
                "trec-deep-learning-passages-2019",
                "trec-deep-learning-passages-2020",
            ]
            if self._ir_datasets is not None:
                for ds_name in standard:
                    try:
                        _ = list(self._ir_datasets.load(ds_name).queries_iter())[:1]
                    except Exception:
                        pass
            if self._pt is None:
                self._lazy_imports()
        except Exception:
            # Preload failures are non-fatal
            pass


    def _autodetect_files(self, paths: List[str]) -> Dict[str, str]:
        """Given a list of file paths, try to detect queries/qrels/corpus by filename patterns."""
        import os
        res: Dict[str, str] = {}
        for p in paths:
            name = os.path.basename(p).lower()
            if "qrel" in name:
                res["qrels_path"] = p
            elif "quer" in name or name.endswith("queries.tsv"):
                res["queries_path"] = p
            elif "corpus" in name or name.endswith(".json") or name.endswith(".jsonl") or name.endswith(".tsv"):
                res["corpus_path"] = p
        return res

    def _autodetect_from_folder(self, folder: str) -> Dict[str, str]:
        import os
        # Prefer BEIR naming
        candidates = {
            'queries_path': [
                os.path.join(folder, 'queries.jsonl'),
                os.path.join(folder, 'queries.json'),
                os.path.join(folder, 'queries.tsv'),
            ],
            'qrels_path': [
                os.path.join(folder, 'qrels', 'test.tsv'),
                os.path.join(folder, 'qrels', 'dev.tsv'),
                os.path.join(folder, 'qrels', 'train.tsv'),
                os.path.join(folder, 'qrels.tsv'),
            ],
            'corpus_path': [
                os.path.join(folder, 'corpus.jsonl'),
                os.path.join(folder, 'corpus.json'),
                os.path.join(folder, 'corpus.tsv'),
            ],
        }
        res: Dict[str, str] = {}
        for key, options in candidates.items():
            for path in options:
                if os.path.exists(path):
                    res[key] = path
                    break
        return res



"""
Incoming: none --- {}
Processing: environment setup --- {1 job: cache path configuration}
Outgoing: env vars --- {PYSERINI_CACHE, HF_HOME, TRANSFORMERS_CACHE, etc.}

Cache Configuration
-------------------
Centralizes cache paths for all models/indexes on Disk-D.
Must be imported BEFORE any ML libraries (transformers, sentence_transformers, pyserini).
"""

import os
from pathlib import Path

# Project root is parent of src/
PROJECT_ROOT = Path(__file__).parent.parent
CACHE_ROOT = PROJECT_ROOT / "cache"

def setup_cache():
    """Set environment variables for all caches to use Disk-D."""
    CACHE_ROOT.mkdir(exist_ok=True)
    
    # Fix FAISS/Java threading crash on Apple Silicon
    os.environ["OMP_NUM_THREADS"] = "1"
    
    # Pyserini index cache
    os.environ["PYSERINI_CACHE"] = str(CACHE_ROOT / "pyserini")
    
    # HuggingFace model cache
    os.environ["HF_HOME"] = str(CACHE_ROOT / "huggingface")
    os.environ["TRANSFORMERS_CACHE"] = str(CACHE_ROOT / "huggingface" / "transformers")
    
    # Sentence Transformers cache
    os.environ["SENTENCE_TRANSFORMERS_HOME"] = str(CACHE_ROOT / "sentence_transformers")
    
    # PyTorch hub cache
    os.environ["TORCH_HOME"] = str(CACHE_ROOT / "torch")
    
    return CACHE_ROOT

# Auto-setup on import
CACHE_ROOT = setup_cache()


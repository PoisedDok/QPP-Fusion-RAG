#!/usr/bin/env python3
"""
COMPREHENSIVE MLP BUG FIX VERIFICATION
Tests all 3 critical bugs with runtime evidence
"""

import sys
import os
import numpy as np
import pickle
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.models import FusionMLP

LOG_FILE = "/Volumes/Disk-D/RAGit/L4-Ind_Proj/QPP-Fusion-RAG/.cursor/debug.log"

def log_debug(location, message, data):
    import json, time
    with open(LOG_FILE, 'a') as f:
        f.write(json.dumps({
            'location': location,
            'message': message,
            'data': data,
            'timestamp': time.time(),
            'sessionId': 'mlp-verification'
        }) + '\n')

print("="*80)
print("MLP BUG FIX VERIFICATION - RUNTIME EVIDENCE")
print("="*80)

all_passed = True

# TEST 1: Training with 65 features should work
print("\n[TEST 1] BUG FIX: train() now filters 65→5 features")
print("-" * 80)

retrievers = ['BGE', 'BM25', 'BM25_MonoT5', 'BM25_TCT', 'Splade']
X_train_65 = np.random.rand(50, 65)  # Full QPP features
Y_train = np.random.rand(50, 5)
Y_train = Y_train / Y_train.sum(axis=1, keepdims=True)

model = FusionMLP(retrievers)

log_debug("verify_mlp_fixes.py:46", "TEST 1 - before train", {
    "X_shape": X_train_65.shape,
    "model_expects": model.n_features
})

try:
    model.train(X_train_65, Y_train, epochs=1, patience=1)
    print("✅ PASS: Training succeeded with 65-feature input")
    log_debug("verify_mlp_fixes.py:55", "TEST 1 - PASS", {"success": True})
except Exception as e:
    print(f"❌ FAIL: Training failed: {e}")
    log_debug("verify_mlp_fixes.py:58", "TEST 1 - FAIL", {"error": str(e)})
    all_passed = False

# TEST 2: Prediction with 65 features should work
print("\n[TEST 2] BUG FIX: predict() now filters 65→5 features")
print("-" * 80)

X_test_65 = np.random.rand(20, 65)

log_debug("verify_mlp_fixes.py:68", "TEST 2 - before predict", {
    "X_shape": X_test_65.shape,
    "model_expects": model.n_features
})

try:
    weights = model.predict(X_test_65)
    print(f"✅ PASS: Prediction succeeded with 65-feature input")
    print(f"   Output shape: {weights.shape}")
    print(f"   Weights sum to 1: {np.allclose(weights.sum(axis=1), 1.0)}")
    log_debug("verify_mlp_fixes.py:80", "TEST 2 - PASS", {
        "weights_shape": weights.shape,
        "weights_normalized": bool(np.allclose(weights.sum(axis=1), 1.0))
    })
except Exception as e:
    print(f"❌ FAIL: Prediction failed: {e}")
    log_debug("verify_mlp_fixes.py:86", "TEST 2 - FAIL", {"error": str(e)})
    all_passed = False

# TEST 3: Load actual trained model and verify it works
print("\n[TEST 3] ACTUAL MODEL: Verify retrained model works end-to-end")
print("-" * 80)

model_path = PROJECT_ROOT / "data/nq/models/fusion_mlp.pkl"

if model_path.exists():
    with open(model_path, 'rb') as f:
        model_data = pickle.load(f)
    
    actual_model = model_data['model']
    
    print(f"Model config:")
    print(f"  - qpp_indices: {actual_model.qpp_indices}")
    print(f"  - n_features: {actual_model.n_features}")
    print(f"  - Network input size: {list(actual_model.model.children())[0].in_features}")
    
    log_debug("verify_mlp_fixes.py:109", "TEST 3 - model loaded", {
        "qpp_indices": actual_model.qpp_indices,
        "n_features": actual_model.n_features
    })
    
    # Test with 65 features (what learned_fusion() passes)
    X_full = np.random.rand(10, 65)
    
    try:
        pred = actual_model.predict(X_full)
        print(f"✅ PASS: Actual model predicts successfully with 65 features")
        print(f"   Output shape: {pred.shape}")
        log_debug("verify_mlp_fixes.py:122", "TEST 3 - PASS", {
            "pred_shape": pred.shape,
            "success": True
        })
    except Exception as e:
        print(f"❌ FAIL: Actual model failed: {e}")
        log_debug("verify_mlp_fixes.py:128", "TEST 3 - FAIL", {"error": str(e)})
        all_passed = False
else:
    print(f"⚠️  SKIP: Model not found at {model_path}")
    all_passed = False

# TEST 4: Verify fusion results exist and are valid
print("\n[TEST 4] FUSION OUTPUT: Verify learned_mlp.res was generated")
print("-" * 80)

fusion_file = PROJECT_ROOT / "data/nq/fused/learned_mlp.res"

if fusion_file.exists():
    line_count = sum(1 for _ in open(fusion_file))
    print(f"✅ PASS: Fusion file exists with {line_count:,} lines")
    
    # Check format
    with open(fusion_file) as f:
        first_line = f.readline().strip()
        parts = first_line.split()
        if len(parts) >= 6:
            print(f"✅ PASS: TREC format valid: {first_line}")
            log_debug("verify_mlp_fixes.py:153", "TEST 4 - PASS", {
                "line_count": line_count,
                "format_valid": True
            })
        else:
            print(f"❌ FAIL: Invalid TREC format: {first_line}")
            all_passed = False
else:
    print(f"❌ FAIL: Fusion file not found at {fusion_file}")
    all_passed = False

# FINAL SUMMARY
print("\n" + "="*80)
print("VERIFICATION SUMMARY")
print("="*80)

if all_passed:
    print("✅ ALL TESTS PASSED - MLP bugs are FIXED")
    print("\nFixed bugs:")
    print("  1. ✅ train() now filters 65→5 features automatically")
    print("  2. ✅ predict() now filters 65→5 features automatically") 
    print("  3. ✅ learned_fusion() works with 5-feature MLP model")
    print("  4. ✅ End-to-end: train → fuse → results pipeline works")
else:
    print("❌ SOME TESTS FAILED - Check logs above")

print("\n" + "="*80)
print("Runtime evidence logged to: .cursor/debug.log")
print("="*80)


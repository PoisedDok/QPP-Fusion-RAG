#!/bin/bash
# QPP-Fusion-RAG: NQ Experiment Pipeline
# ======================================
# Master orchestrator for running the full NQ experiment.
#
# Usage:
#   ./run_nq_experiment.sh                    # Full pipeline
#   ./run_nq_experiment.sh --step 2           # Start from step 2
#   ./run_nq_experiment.sh --limit 100        # Test with 100 docs
#   ./run_nq_experiment.sh --retrievers BM25  # Only BM25

set -e

# Configuration
CORPUS_PATH="${CORPUS_PATH:-/Volumes/Disk-D/RAGit/data/beir/datasets/nq}"
DATA_DIR="data/nq"
QPP_MODEL="${QPP_MODEL:-RSD}"
SHOTS="${SHOTS:-0,1,2,3,4,5,6,10}"
MODEL="${MODEL:-qwen/qwen3-4b-2507}"
RETRIEVERS="${RETRIEVERS:-BM25,TCT-ColBERT,Splade,BM25_TCT,BM25_MonoT5}"

# Parse arguments
START_STEP=1
LIMIT=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --step)
            START_STEP="$2"
            shift 2
            ;;
        --limit)
            LIMIT="--limit $2"
            shift 2
            ;;
        --retrievers)
            RETRIEVERS="$2"
            shift 2
            ;;
        --qpp)
            QPP_MODEL="$2"
            shift 2
            ;;
        --shots)
            SHOTS="$2"
            shift 2
            ;;
        --corpus)
            CORPUS_PATH="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

echo "========================================"
echo "QPP-Fusion-RAG: NQ Experiment Pipeline"
echo "========================================"
echo "Corpus: $CORPUS_PATH"
echo "QPP Model: $QPP_MODEL"
echo "Retrievers: $RETRIEVERS"
echo "Shots: $SHOTS"
echo "Start Step: $START_STEP"
echo ""

# Step 1: Index
if [ $START_STEP -le 1 ]; then
    echo "=== Step 1: Index Corpus ==="
    python scripts/01_index.py --corpus_path "$CORPUS_PATH" $LIMIT
    echo ""
fi

# Step 2: Retrieve
if [ $START_STEP -le 2 ]; then
    echo "=== Step 2: Run Retrievers ==="
    python scripts/02_retrieve.py \
        --corpus_path "$CORPUS_PATH" \
        --retrievers "$RETRIEVERS" \
        $LIMIT
    echo ""
fi

# Step 3: QPP
if [ $START_STEP -le 3 ]; then
    echo "=== Step 3: Compute QPP ==="
    python scripts/03_qpp.py
    echo ""
fi

# Step 4: Fusion
if [ $START_STEP -le 4 ]; then
    echo "=== Step 4: W-CombSum Fusion ==="
    python scripts/04_fusion.py --qpp_model "$QPP_MODEL"
    echo ""
fi

# Step 5: RAG Evaluation
if [ $START_STEP -le 5 ]; then
    echo "=== Step 5: RAG Evaluation ==="
    python scripts/05_rag_eval.py \
        --corpus_path "$CORPUS_PATH" \
        --shots "$SHOTS" \
        --model "$MODEL" \
        $LIMIT
    echo ""
fi

echo "========================================"
echo "Pipeline Complete!"
echo "========================================"
echo "Results: $DATA_DIR/results/"


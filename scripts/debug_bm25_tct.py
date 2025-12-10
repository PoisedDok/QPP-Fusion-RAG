#!/usr/bin/env python3
"""
BM25_TCT Diagnostic Script
--------------------------
Tests the two-stage pipeline to identify the failure point.
"""

import sys
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
LOG_FILE = PROJECT_ROOT / ".cursor" / "debug.log"

def log(location: str, message: str, data: dict = None, hypothesis: str = ""):
    """Write NDJSON log."""
    import time
    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    entry = {
        "timestamp": time.time(),
        "location": location,
        "message": message,
        "data": data or {},
        "hypothesisId": hypothesis
    }
    with open(LOG_FILE, "a") as f:
        f.write(json.dumps(entry) + "\n")


def main():
    from src.config import config, ensure_pyterrier_init
    from src.data_utils import load_qrels, load_queries
    
    pt = ensure_pyterrier_init()
    
    dataset = "nq"
    beir_path = config.get_beir_path(dataset)
    index_path = str(config.project_root / "data" / dataset / "index" / "pyterrier")
    qrels_path = config.get_qrels_path(dataset)
    
    # Load test data
    qrels = load_qrels(qrels_path)
    queries = load_queries(beir_path)
    
    # Sample 10 queries that have relevant documents
    test_qids = [qid for qid in list(queries.keys())[:50] if qid in qrels][:10]
    test_queries = {qid: queries[qid] for qid in test_qids}
    
    log("main:30", "Starting diagnostic", {"n_queries": len(test_queries), "qids": test_qids}, "H0")
    
    # =========================================================================
    # H1: BM25 first-stage recall problem
    # =========================================================================
    print("\n=== H1: BM25 First-Stage Recall ===")
    index = pt.IndexFactory.of(index_path)
    bm25 = pt.BatchRetrieve(
        index, wmodel="BM25", num_results=100,
        metadata=["docno"],
        controls={"bm25.k_1": "0.9", "bm25.b": "0.4"}
    )
    
    import pandas as pd
    query_df = pd.DataFrame([{"qid": qid, "query": text} for qid, text in test_queries.items()])
    bm25_results = bm25.transform(query_df)
    
    bm25_recall = {}
    for qid in test_qids:
        relevant = set(qrels.get(qid, {}).keys())
        retrieved = set(bm25_results[bm25_results["qid"] == qid]["docno"].tolist())
        found = relevant & retrieved
        recall = len(found) / len(relevant) if relevant else 0
        bm25_recall[qid] = {"recall": recall, "found": len(found), "total_rel": len(relevant)}
        print(f"  {qid}: recall@100 = {recall:.2%} ({len(found)}/{len(relevant)})")
    
    avg_recall = sum(r["recall"] for r in bm25_recall.values()) / len(bm25_recall)
    log("h1_check:60", "BM25 recall@100", {"avg_recall": avg_recall, "per_query": bm25_recall}, "H1")
    print(f"  Avg Recall@100: {avg_recall:.2%}")
    
    # =========================================================================
    # H2: TCT-ColBERT scoring problem
    # =========================================================================
    print("\n=== H2: TCT-ColBERT Scoring ===")
    import pyterrier_dr as dr
    from src.config import get_device
    
    device = get_device()
    tct_reranker = dr.TctColBert.hnp(batch_size=32, verbose=False, device=device).text_scorer()
    
    # Load corpus texts
    corpus_file = beir_path / "corpus.jsonl"
    doc_texts = {}
    print(f"  Loading corpus texts...")
    
    unique_docs = bm25_results["docno"].unique().tolist()
    corpus_lookup = {}
    with open(corpus_file, 'r') as f:
        for line in f:
            doc = json.loads(line)
            if doc["_id"] in unique_docs:
                corpus_lookup[doc["_id"]] = (doc.get("title", "") + " " + doc.get("text", "")).strip()
    
    bm25_with_text = bm25_results.copy()
    bm25_with_text["text"] = bm25_with_text["docno"].apply(lambda x: corpus_lookup.get(x, ""))
    
    # Check for empty texts
    empty_count = (bm25_with_text["text"] == "").sum()
    log("h2_text:90", "Text loading", {"empty_texts": int(empty_count), "total": len(bm25_with_text)}, "H2")
    print(f"  Empty texts: {empty_count}/{len(bm25_with_text)}")
    
    # TCT reranking
    print(f"  Running TCT-ColBERT reranking...")
    reranked = tct_reranker.transform(bm25_with_text)
    
    # =========================================================================
    # H3: Score distribution analysis
    # =========================================================================
    print("\n=== H3: Score Distribution Analysis ===")
    
    tct_scores_relevant = []
    tct_scores_nonrel = []
    
    for qid in test_qids:
        relevant = set(qrels.get(qid, {}).keys())
        qid_results = reranked[reranked["qid"] == qid]
        
        for _, row in qid_results.iterrows():
            if row["docno"] in relevant:
                tct_scores_relevant.append(row["score"])
            else:
                tct_scores_nonrel.append(row["score"])
    
    if tct_scores_relevant:
        avg_rel = sum(tct_scores_relevant) / len(tct_scores_relevant)
        avg_nonrel = sum(tct_scores_nonrel) / len(tct_scores_nonrel) if tct_scores_nonrel else 0
        print(f"  Relevant doc avg score: {avg_rel:.4f}")
        print(f"  Non-relevant doc avg score: {avg_nonrel:.4f}")
        print(f"  Score gap: {avg_rel - avg_nonrel:.4f}")
        log("h3_scores:120", "Score distribution", {
            "rel_avg": avg_rel, "nonrel_avg": avg_nonrel, 
            "n_rel": len(tct_scores_relevant), "n_nonrel": len(tct_scores_nonrel)
        }, "H3")
    else:
        print(f"  No relevant documents found in top-100!")
        log("h3_scores:125", "No relevant docs found", {}, "H3")
    
    # =========================================================================
    # H4: Rank comparison (BM25 vs TCT)
    # =========================================================================
    print("\n=== H4: Rank Comparison (BM25 vs TCT) ===")
    
    for qid in test_qids[:3]:  # First 3 queries
        relevant = set(qrels.get(qid, {}).keys())
        
        # BM25 ranks
        bm25_qid = bm25_results[bm25_results["qid"] == qid].sort_values("score", ascending=False)
        bm25_ranks = {row["docno"]: i+1 for i, (_, row) in enumerate(bm25_qid.iterrows())}
        
        # TCT ranks
        tct_qid = reranked[reranked["qid"] == qid].sort_values("score", ascending=False)
        tct_ranks = {row["docno"]: i+1 for i, (_, row) in enumerate(tct_qid.iterrows())}
        
        print(f"\n  Query {qid}:")
        found_rel = [d for d in relevant if d in bm25_ranks]
        for doc_id in found_rel[:3]:  # First 3 relevant docs
            bm25_r = bm25_ranks.get(doc_id, ">100")
            tct_r = tct_ranks.get(doc_id, ">100")
            print(f"    {doc_id}: BM25 rank={bm25_r}, TCT rank={tct_r}")
    
    # =========================================================================
    # H5: Compare with standalone BM25 evaluation
    # =========================================================================
    print("\n=== H5: nDCG@10 Comparison ===")
    import ir_measures
    from ir_measures import nDCG
    
    # BM25 nDCG@10
    bm25_run = {}
    for qid in test_qids:
        qid_results = bm25_results[bm25_results["qid"] == qid].nlargest(10, "score")
        bm25_run[qid] = {row["docno"]: row["score"] for _, row in qid_results.iterrows()}
    
    qrels_ir = {qid: {d: r for d, r in rels.items()} for qid, rels in qrels.items() if qid in test_qids}
    bm25_ndcg = list(ir_measures.iter_calc([nDCG@10], qrels_ir, bm25_run))
    bm25_avg = sum(r.value for r in bm25_ndcg) / len(bm25_ndcg)
    
    # TCT nDCG@10
    tct_run = {}
    for qid in test_qids:
        qid_results = reranked[reranked["qid"] == qid].nlargest(10, "score")
        tct_run[qid] = {row["docno"]: row["score"] for _, row in qid_results.iterrows()}
    
    tct_ndcg = list(ir_measures.iter_calc([nDCG@10], qrels_ir, tct_run))
    tct_avg = sum(r.value for r in tct_ndcg) / len(tct_ndcg)
    
    print(f"  BM25 nDCG@10: {bm25_avg:.4f}")
    print(f"  TCT nDCG@10:  {tct_avg:.4f}")
    print(f"  Change: {(tct_avg - bm25_avg):.4f} ({(tct_avg - bm25_avg)/bm25_avg*100:+.1f}%)")
    
    log("h5_ndcg:180", "nDCG comparison", {
        "bm25_ndcg": bm25_avg, "tct_ndcg": tct_avg, 
        "change": tct_avg - bm25_avg
    }, "H5")
    
    # Per-query comparison
    print("\n  Per-query nDCG@10:")
    bm25_by_qid = {r.query_id: r.value for r in bm25_ndcg}
    tct_by_qid = {r.query_id: r.value for r in tct_ndcg}
    
    improvements = 0
    degradations = 0
    for qid in test_qids:
        bm25_v = bm25_by_qid.get(qid, 0)
        tct_v = tct_by_qid.get(qid, 0)
        delta = tct_v - bm25_v
        if delta > 0.01:
            improvements += 1
        elif delta < -0.01:
            degradations += 1
        print(f"    {qid}: BM25={bm25_v:.4f} -> TCT={tct_v:.4f} ({delta:+.4f})")
    
    print(f"\n  Improvements: {improvements}, Degradations: {degradations}")
    log("h5_perquery:200", "Per-query analysis", {
        "improvements": improvements, "degradations": degradations
    }, "H5")
    
    print("\n=== Diagnostic Complete ===")
    print(f"Log file: {LOG_FILE}")


if __name__ == "__main__":
    main()

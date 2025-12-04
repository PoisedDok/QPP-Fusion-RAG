#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CombMNZ and Weighted CombMNZ evaluation script (PyTerrier)
Implements:
    CombMNZ(d,q) = (# of rankers where d appears) * sum_{i : d in L_i} S_{L_i}^{norm}(d,q)
    W-CombMNZ(d,q) = (# of rankers where d appears) * sum_{i : d in L_i} w_i(q) * S_{L_i}^{norm}(d,q)

Usage:
    python combmnz_eval.py --res_path ./runs_norm --qpp_path ./qpp --qrels qrels.txt --topics topics.txt --strategy combmnz
    python combmnz_eval.py --res_path ./runs_norm --qpp_path ./qpp --qrels qrels.txt --topics topics.txt --strategy wcombmnz
"""

import os
import argparse
import pandas as pd
from collections import defaultdict
import pyterrier as pt

# if not pt.started():
#     pt.init()

# QPP model index -> name mapping (same as provided)
model_name_dict={0:"SMV",1:"Sigma_max",2:"Sigma(%)",3:"NQC",4:"UEF",5:"RSD",
                 6:"QPP-PRP",7:"WIG",8:"SCNQC",9:"QV-NQC",10:"DM",
                 11:"NQA-QPP",12:"BERTQPP"}


def load_qpp_estimates(qpp_path):
    """
    Loads .znorm.qpp files from qpp_path.
    Expects files named like <ranker>.res.znorm.qpp (as in your original loader).
    Returns: qpp_data[qid][ranker] = [qpp_col0, qpp_col1, ..., qpp_colN]
    """
    qpp_data = defaultdict(dict)
    # collect .znorm.qpp files
    files = [os.path.join(qpp_path, f) for f in os.listdir(qpp_path) if f.endswith(".mmnorm.qpp")]
    for f in files:
        # Keep the same logic as your loader: remove suffix ".res.znorm.qpp" to get ranker name
        ranker = os.path.basename(f).replace(".res.mmnorm.qpp","")
        with open(f, "r") as fin:
            for line in fin:
                parts = line.strip().split("\t")
                if len(parts) < 2:
                    continue
                qid = parts[0]
                scores = [float(x) for x in parts[1:]]  # QPP columns
                qpp_data[qid][ranker] = scores
    return qpp_data


def load_runs(res_path):
    """
    Loads .norm.res run files into a dict of pandas DataFrames.
    Converts qid column to string to match qpp_data keys.
    Expects filenames like <ranker>.100.norm.res or <ranker>.norm.res (we use suffix detection).
    """
    runs = {}
    files = [f for f in os.listdir(res_path) if f.endswith(".norm.res")]
    for f in files:
        ranker = f.replace(".norm.res","")
        df = pd.read_csv(os.path.join(res_path, f), sep=r"\s+",
                         names=["qid","iter","docno","rank","score","runid"],
                         dtype={"qid":str, "docno":str})
        # ensure qid is string (so it matches qpp_data keys)
        df["qid"] = df["qid"].astype(str)
        runs[ranker] = df
    return runs


def build_combmnz_run(runs, qpp_data=None, qpp_method_index=None):
    """
    Build CombMNZ or Weighted CombMNZ run (as DataFrame) similar to your build_combsum_run.
    If qpp_data and qpp_method_index provided -> weighted variant (W-CombMNZ) is computed.
    Returns a pandas DataFrame with columns: ["qid","iter","docno","rank","score","runid"]
    """
    records = []
    rankers = list(runs.keys())
    # collect all unique qids across rankers
    all_qids = set()
    for r in rankers:
        all_qids.update(runs[r]["qid"].unique())
    all_qids = sorted(all_qids, key=lambda x: int(x) if x.isdigit() else x)

    for qid in all_qids:
        doc_scores = defaultdict(float)
        doc_counts = defaultdict(int)

        for ranker in rankers:
            df_q = runs[ranker][runs[ranker]["qid"] == qid]
            if df_q.empty:
                continue

            # determine weight (1.0 default for unweighted CombMNZ)
            weight = 1.0
            if qpp_data is not None and qpp_method_index is not None:
                # try to fetch weight for this qid and ranker; fallback to 1.0 (uniform) if missing
                try:
                    weight = qpp_data[qid][ranker][qpp_method_index]
                except KeyError:
                    weight = 1.0  # Missing QPP → uniform weight, not zero
                except Exception:
                    weight = 1.0

            # accumulate weighted sums and counts
            for _, row in df_q.iterrows():
                doc_scores[row["docno"]] += weight * row["score"]
                doc_counts[row["docno"]] += 1

        # compute CombMNZ score = count * sum_scores (or count * weighted_sum)
        if doc_scores:
            # sort by fused score
            fused_list = []
            for docno, ssum in doc_scores.items():
                cnt = doc_counts.get(docno, 0)
                fused_score = cnt * ssum
                fused_list.append((docno, fused_score))
            fused_list.sort(key=lambda x: x[1], reverse=True)

            # append to records
            runid = "W-CombMNZ" if qpp_data else "CombMNZ"
            for rank, (docno, score) in enumerate(fused_list, start=1):
                records.append([qid, "Q0", docno, rank, score, runid])

    return pd.DataFrame(records, columns=["qid","iter","docno","rank","score","runid"])


def main():
    parser = argparse.ArgumentParser(description="CombMNZ / W-CombMNZ evaluation (PyTerrier)")
    parser.add_argument("--res_path", type=str, required=True, help="Path to directory containing .norm.res run files")
    parser.add_argument("--qpp_path", type=str, required=True, help="Path to directory containing .znorm.qpp files")
    parser.add_argument("--qrels", type=str, required=True, help="Path to qrels file")
    parser.add_argument("--topics", type=str, required=True, help="Path to topics file (qid \\t query)")
    parser.add_argument("--metrics", type=str, nargs='+', default=["map","ndcg@10","recip_rank"], help="Evaluation metrics")
    parser.add_argument("--strategy", type=str, choices=["combmnz","wcombmnz"], required=True,
                        help="Choose fusion strategy: combmnz | wcombmnz")
    args = parser.parse_args()

    # load runs and qpp data
    runs = load_runs(args.res_path)
    qpp_data = load_qpp_estimates(args.qpp_path)

    # read qrels and topics (topics qid as string)
    qrels = pt.io.read_qrels(args.qrels)
    topics = pd.read_csv(args.topics, sep="\t", names=["qid","query"]).astype({'qid':'str'})

    # detect how many QPP methods per ranker (columns)
    # safe navigation in case qpp_data is empty
    try:
        sample_scores = next(iter(next(iter(qpp_data.values())).values()))
        num_qpp_methods = len(sample_scores)
    except StopIteration:
        num_qpp_methods = 0
    print(f"Detected {num_qpp_methods} QPP methods (columns) per ranker")

    # prepare measures
    measures = []
    for m in args.metrics:
        if m.lower() == "map":
            measures.append(pt.measures.AP(rel=2))
        elif m.lower().startswith("ndcg"):
            if "@" in m:
                k = int(m.split("@")[1])
                measures.append(pt.measures.nDCG(cutoff=k))
            else:
                measures.append(pt.measures.nDCG())
        elif m.lower() == "recip_rank" or m.lower() == "rr":
            measures.append(pt.measures.RR())
        else:
            # fallback: try to support common shorthand
            if m.lower() == "p5":
                measures.append(pt.measures.P(cutoff=5))
            elif m.lower() == "p10":
                measures.append(pt.measures.P(cutoff=10))
            else:
                raise ValueError(f"Unsupported metric: {m}")

    all_runs, all_names = [], []

    # If standard CombMNZ
    if args.strategy == "combmnz":
        df = build_combmnz_run(runs)
        all_runs.append(df)
        all_names.append("CombMNZ")

    # If weighted: build one fused run per QPP method (column)
    elif args.strategy == "wcombmnz":
        if num_qpp_methods == 0:
            raise ValueError("No QPP methods detected in qpp_path. Ensure .znorm.qpp files exist and are non-empty.")
        for i in range(num_qpp_methods):
            df = build_combmnz_run(runs, qpp_data=qpp_data, qpp_method_index=i)
            model_name = model_name_dict.get(i, f"QPP-{i}")
            all_runs.append(df)
            all_names.append(f"W-CombMNZ-QPP-{model_name}")

    # Always add baseline runs (original rankers)
    # convert runs dict -> dataframes (PyTerrier expects DataFrames in the experiment)
    for ranker, df in runs.items():
        # For naming consistency use the ranker key
        all_runs.append(df)
        all_names.append(ranker)

    # Run evaluation
    results = pt.Experiment(
        all_runs,
        qrels=qrels,
        topics=topics,
        eval_metrics=measures,
        names=all_names
    )

    print(results)


if __name__ == "__main__":
    main()


# python3 eval-precise-qpp/CombMNZ.py  --res_path lucene-msmarco/data/runs/2019/RL_norm --qrels eval-precise-qpp/data/2019.qrels --topics eval-precise-qpp/data/2019.queries  --strategy combmnz --qpp_path lucene-msmarco/data/runs/2019/norm_qpp2 

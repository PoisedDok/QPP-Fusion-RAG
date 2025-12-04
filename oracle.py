import pyterrier as pt
import pandas as pd
import os
import argparse

def load_runs(res_path):
    """
    Load all ranker runs as DataFrames
    Returns: dict {ranker: df}
    """
    runs = {}
    files = [f for f in os.listdir(res_path) if f.endswith(".norm.res")] #minmax
    for f in files:
        ranker = f.replace(".norm.res","")
        df = pd.read_csv(os.path.join(res_path, f), sep=r"\s+", 
                         names=["qid","iter","docno","rank","score","runid"])
        runs[ranker] = df
    return runs

def compute_oracle_run(runs, topics, qrels, metric="map"):
    """
    Build oracle adaptive run: for each query, select the ranker 
    that achieves the highest effectiveness (AP/nDCG@k/etc.).
    """
    records = []

    # Collect all query IDs
    all_qids = set()
    for df in runs.values():
        all_qids.update(df["qid"].unique())

    # Select measure
    if metric.lower() == "map":
        measure = pt.measures.AP(rel=2)
    elif "ndcg" in metric.lower():
        if "@" in metric:
            k = int(metric.split("@")[1])
            measure = pt.measures.nDCG(k=k)
        else:
            measure = pt.measures.nDCG()
    elif metric.lower() == "recip_rank":
        measure = pt.measures.RR()
    else:
        raise ValueError(f"Unsupported metric: {metric}")

    # Loop over queries
    for qid in all_qids:
        best_score = -1
        best_df = None

        for ranker, df in runs.items():
            df_q = df[df["qid"] == int(qid)]
            if df_q.empty:
                continue

            res = pt.Experiment(
                [df_q],
                topics=topics,
                qrels=qrels,
                eval_metrics=[measure]
            )
            # print("res",res)
            # print("-----",measure)
            score = res.iloc[0, 0]  # single measure
            score = float(res.iloc[0]["AP(rel=2)"]) #["AP(rel=2)"]). #chnge ["nDCG"]

            if score > best_score:
                best_score = score
                best_df = df_q

        if best_df is not None:
            records.append(best_df)

    oracle_df = pd.concat(records)
    oracle_df = oracle_df.sort_values(["qid", "score"], ascending=[True, False])
    oracle_df["rank"] = oracle_df.groupby("qid").cumcount() + 1
    return oracle_df

def main():
    parser = argparse.ArgumentParser(description="Oracle Setup Evaluation")
    parser.add_argument("--res_path", type=str, required=True, help="Path to folder with .res files")
    parser.add_argument("--qrels", type=str, required=True, help="Path to qrels file")
    parser.add_argument("--topics", type=str, required=True, help="Path to topics file (CSV: qid,query)")
    parser.add_argument("--metric", type=str, default="map", help="Metric: map | ndcg@10 | recip_rank")
    args = parser.parse_args()

    runs = load_runs(args.res_path)
    qrels = pt.io.read_qrels(args.qrels)
    topics = pd.read_csv(args.topics, sep="\t", names=["qid","query"]).astype({'qid':'str'})

    oracle_df = compute_oracle_run(runs, topics, qrels, metric=args.metric)

    # Evaluate oracle run
    if args.metric.lower() == "map":
        measures = [pt.measures.AP(rel=2)]
    else:
        measures = [pt.measures.nDCG(cutoff=10)]

    results = pt.Experiment(
        [oracle_df],
        qrels=qrels,
        topics=topics,
        eval_metrics=measures,
        names=["Oracle"]
    )

    print("\n=== Oracle Performance ===")
    print(results)

if __name__ == "__main__":
    # if not pt.started():
    #     pt.init()
    main()

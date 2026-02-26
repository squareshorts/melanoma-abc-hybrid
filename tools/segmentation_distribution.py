# save as tools/segmentation_distribution.py
import pandas as pd
import numpy as np

def summarize(series):
    return {
        "mean": float(np.mean(series)),
        "std": float(np.std(series, ddof=1)),
        "median": float(np.median(series)),
        "p10": float(np.percentile(series, 10)),
        "p25": float(np.percentile(series, 25)),
        "p75": float(np.percentile(series, 75)),
        "p90": float(np.percentile(series, 90)),
    }

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--full_csv", required=True, help="per-image segmentation metrics CSV, must include image_id,dice,iou,method")
    ap.add_argument("--subset_ids", required=False, help="txt/csv with image_ids for subset")
    ap.add_argument("--method", required=True, help="LevelSet or OtsuBaseline")
    args = ap.parse_args()

    df = pd.read_csv(args.full_csv)
    df = df[df["method"] == args.method].copy()

    print("FULL", args.method, summarize(df["dice"].values), "N=", len(df))

    if args.subset_ids:
        sid = pd.read_csv(args.subset_ids, header=None).iloc[:,0].astype(str).tolist()
        sub = df[df["image_id"].astype(str).isin(sid)]
        print("SUBSET", args.method, summarize(sub["dice"].values), "N=", len(sub))
        # percentile of subset median in full
        full_sorted = np.sort(df["dice"].values)
        sub_med = np.median(sub["dice"].values)
        pct = 100.0 * (full_sorted <= sub_med).mean()
        print("subset median percentile in full:", pct)
# tools/make_stratified_subset.py
import pandas as pd
import numpy as np

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--full_csv", required=True)
    ap.add_argument("--method", required=True)
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--bins", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out_csv", required=True)
    args = ap.parse_args()

    df = pd.read_csv(args.full_csv)
    df = df[df["method"] == args.method].copy()
    df["qbin"] = pd.qcut(df["dice"], q=args.bins, labels=False, duplicates="drop")

    rng = np.random.default_rng(args.seed)
    per_bin = args.n // df["qbin"].nunique()
    picks = []
    for b in sorted(df["qbin"].unique()):
        sub = df[df["qbin"] == b]
        k = min(per_bin, len(sub))
        idx = rng.choice(sub.index.values, size=k, replace=False)
        picks.extend(idx.tolist())

    out = df.loc[picks, ["image_id"]].drop_duplicates()
    out.to_csv(args.out_csv, index=False, header=False)
    print("Wrote", args.out_csv, "N=", len(out))
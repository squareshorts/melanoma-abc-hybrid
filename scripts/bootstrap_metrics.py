# tools/bootstrap_metrics.py
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss

def bootstrap_ci(y, p, metric_fn, n=1000, seed=42):
    rng = np.random.default_rng(seed)
    y = np.asarray(y).astype(int)
    p = np.asarray(p).astype(float)
    vals = []
    N = len(y)
    for _ in range(n):
        idx = rng.integers(0, N, size=N)
        yy = y[idx]
        pp = p[idx]
        # guard: skip degenerate resamples for AUC
        if metric_fn in (roc_auc_score, average_precision_score):
            if len(np.unique(yy)) < 2:
                continue
        vals.append(metric_fn(yy, pp))
    vals = np.array(vals, dtype=float)
    return float(np.mean(vals)), float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5)), len(vals)

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred_csv", required=True, help="CSV with columns y,p")
    ap.add_argument("--out_csv", required=True)
    ap.add_argument("--y_col", default="y")
    ap.add_argument("--p_col", default="p")
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    df = pd.read_csv(args.pred_csv)
    y = df[args.y_col].values
    p = df[args.p_col].values

    rows = []
    for name, fn in [("roc_auc", roc_auc_score), ("pr_auc", average_precision_score), ("brier", brier_score_loss)]:
        mean, lo, hi, m = bootstrap_ci(y, p, fn, n=args.n, seed=args.seed)
        rows.append({"metric": name, "mean": mean, "ci_lo": lo, "ci_hi": hi, "n_eff": m})
    pd.DataFrame(rows).to_csv(args.out_csv, index=False)
    print("Wrote", args.out_csv)
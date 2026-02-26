# tools/calibration_metrics.py
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss

def ece(y, p, n_bins=10):
    y = np.asarray(y).astype(int)
    p = np.asarray(p).astype(float)
    bins = np.linspace(0, 1, n_bins + 1)
    ece_val = 0.0
    N = len(y)
    for i in range(n_bins):
        lo, hi = bins[i], bins[i+1]
        mask = (p >= lo) & (p < hi) if i < n_bins-1 else (p >= lo) & (p <= hi)
        if mask.sum() == 0:
            continue
        acc = y[mask].mean()
        conf = p[mask].mean()
        ece_val += (mask.sum() / N) * abs(acc - conf)
    return float(ece_val)

def calib_slope_intercept(y, p):
    # Fit logistic regression of y on logit(p): y ~ a + b*logit(p)
    y = np.asarray(y).astype(int)
    p = np.clip(np.asarray(p).astype(float), 1e-6, 1 - 1e-6)
    logit = np.log(p / (1 - p)).reshape(-1, 1)
    lr = LogisticRegression(solver="lbfgs")
    lr.fit(logit, y)
    intercept = float(lr.intercept_[0])
    slope = float(lr.coef_[0][0])
    return intercept, slope

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred_csv", required=True, help="CSV with columns y,p")
    ap.add_argument("--out_csv", required=True)
    ap.add_argument("--y_col", default="y")
    ap.add_argument("--p_col", default="p")
    ap.add_argument("--bins", type=int, default=10)
    args = ap.parse_args()

    df = pd.read_csv(args.pred_csv)
    y = df[args.y_col].values
    p = df[args.p_col].values

    out = {}
    out["brier"] = float(brier_score_loss(y, p))
    out["ece"] = ece(y, p, n_bins=args.bins)
    intercept, slope = calib_slope_intercept(y, p)
    out["calib_intercept"] = intercept
    out["calib_slope"] = slope

    pd.DataFrame([out]).to_csv(args.out_csv, index=False)
    print("Wrote", args.out_csv)
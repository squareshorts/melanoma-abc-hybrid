# save as tools/decision_curve.py
import numpy as np
import pandas as pd

def net_benefit(y_true, p, thr):
    # thr in (0,1)
    y = np.asarray(y_true).astype(int)
    p = np.asarray(p).astype(float)
    pred = (p >= thr).astype(int)
    tp = ((pred == 1) & (y == 1)).sum()
    fp = ((pred == 1) & (y == 0)).sum()
    n = len(y)
    w = thr / (1 - thr)
    return (tp / n) - (fp / n) * w

def decision_curve(df, y_col="y", p_col="p", thr_grid=None):
    if thr_grid is None:
        thr_grid = np.linspace(0.01, 0.99, 99)
    y = df[y_col].values
    p = df[p_col].values
    prev = y.mean()
    out = []
    for t in thr_grid:
        nb_model = net_benefit(y, p, t)
        nb_all = prev - (1 - prev) * (t / (1 - t))  # treat-all
        nb_none = 0.0
        out.append((t, nb_model, nb_all, nb_none))
    return pd.DataFrame(out, columns=["threshold", "net_benefit_model", "net_benefit_all", "net_benefit_none"])

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_csv", required=True, help="CSV with columns y,p")
    ap.add_argument("--out_csv", required=True)
    ap.add_argument("--y_col", default="y")
    ap.add_argument("--p_col", default="p")
    args = ap.parse_args()

    df = pd.read_csv(args.in_csv)
    dca = decision_curve(df, y_col=args.y_col, p_col=args.p_col)
    dca.to_csv(args.out_csv, index=False)
    print("Wrote", args.out_csv)
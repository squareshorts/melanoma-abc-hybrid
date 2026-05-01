import os
import numpy as np
import pandas as pd


def decision_curve(y, p, thresholds):
    y = np.array(y)
    p = np.array(p)

    N = len(y)
    out = []

    for t in thresholds:
        preds = (p >= t).astype(int)

        TP = ((preds == 1) & (y == 1)).sum()
        FP = ((preds == 1) & (y == 0)).sum()

        nb = (TP / N) - (FP / N) * (t / (1 - t))
        out.append({"threshold": t, "net_benefit": nb})

    return pd.DataFrame(out)


def run_one(pred_path, tag):
    df = pd.read_csv(pred_path)
    y = df["y_true"].astype(int).values
    p = df["y_prob"].astype(float).values

    thresholds = np.linspace(0.02, 0.50, 49)
    dca = decision_curve(y, p, thresholds)

    out_path = f"results/audit/dca_isic_{tag}.csv"
    dca.to_csv(out_path, index=False)
    print("Wrote", out_path)


def main():
    os.makedirs("results/audit", exist_ok=True)

    files = {
        "deep_raw": "results/audit/preds_isic_deep_raw.csv",
        "deep_platt": "results/audit/preds_isic_deep_platt.csv",
        "deep_isotonic": "results/audit/preds_isic_deep_isotonic.csv",
        "hyb_raw": "results/audit/preds_isic_hybrid_raw.csv",
        "hyb_platt": "results/audit/preds_isic_hybrid_platt.csv",
        "hyb_isotonic": "results/audit/preds_isic_hybrid_isotonic.csv",
    }

    for tag, path in files.items():
        run_one(path, tag)


if __name__ == "__main__":
    main()

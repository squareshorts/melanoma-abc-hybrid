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


def run_one(path, tag):
    df = pd.read_csv(path)
    y = df["y_true"].values
    p = df["y_prob"].values

    thresholds = np.linspace(0.01, 0.99, 99)
    dca = decision_curve(y, p, thresholds)
    dca.to_csv(f"results/audit/dca_ham_{tag}.csv", index=False)
    return dca


def main():
    os.makedirs("results/audit", exist_ok=True)

    files = {
        "deep_raw": "results/audit/preds_ham_deep_raw.csv",
        "deep_platt": "results/audit/preds_ham_deep_platt.csv",
        "deep_isotonic": "results/audit/preds_ham_deep_isotonic.csv",
        "hyb_raw": "results/audit/preds_ham_hybrid_raw.csv",
        "hyb_platt": "results/audit/preds_ham_hybrid_platt.csv",
        "hyb_isotonic": "results/audit/preds_ham_hybrid_isotonic.csv",
    }

    for tag, path in files.items():
        run_one(path, tag)

    print("Wrote calibrated DCA curves for HAM.")


if __name__ == "__main__":
    main()
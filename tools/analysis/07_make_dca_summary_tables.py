import os
import pandas as pd

RANGES = [
    ("0.10-0.50", 0.10, 0.50),
    ("0.10-0.70", 0.10, 0.70),
    ("0.10-0.90", 0.10, 0.90),
]

def load_dca(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    # expected columns: threshold, net_benefit (plus maybe treat_all/treat_none, ignore)
    if "threshold" not in df.columns:
        raise ValueError(f"{path}: missing 'threshold'")
    if "net_benefit" not in df.columns:
        raise ValueError(f"{path}: missing 'net_benefit'")
    df = df[["threshold", "net_benefit"]].copy()
    df["threshold"] = df["threshold"].astype(float)
    df["net_benefit"] = df["net_benefit"].astype(float)
    return df.sort_values("threshold").reset_index(drop=True)

def delta_stats(df_a: pd.DataFrame, df_b: pd.DataFrame, lo: float, hi: float):
    m = df_a.merge(df_b, on="threshold", suffixes=("_a", "_b"))
    w = m[(m["threshold"] >= lo) & (m["threshold"] <= hi)].copy()
    w["delta"] = w["net_benefit_a"] - w["net_benefit_b"]
    mean_delta = float(w["delta"].mean())
    j = int(w["delta"].idxmax())
    r = w.loc[j]
    return mean_delta, float(r["delta"]), float(r["threshold"])

def model_area(df: pd.DataFrame, lo: float, hi: float):
    w = df[(df["threshold"] >= lo) & (df["threshold"] <= hi)].copy()
    # simple mean NB over grid (your DCA is already on a discrete threshold grid)
    return float(w["net_benefit"].mean())

def summarize_one(prefix: str, deep_paths: dict, hyb_paths: dict):
    rows = []

    def add_block(model_family: str, paths: dict):
        raw = load_dca(paths["raw"])
        platt = load_dca(paths["platt"])
        iso = load_dca(paths["isotonic"])

        for label, lo, hi in RANGES:
            rows.append({
                "dataset": prefix,
                "model_family": model_family,
                "range": label,
                "mean_NB_raw": model_area(raw, lo, hi),
                "mean_NB_platt": model_area(platt, lo, hi),
                "mean_NB_isotonic": model_area(iso, lo, hi),
            })

            mean_d, max_d, thr = delta_stats(platt, raw, lo, hi)
            rows.append({
                "dataset": prefix,
                "model_family": model_family,
                "range": label,
                "comparison": "platt_minus_raw",
                "mean_delta_NB": mean_d,
                "max_delta_NB": max_d,
                "thr_at_max_delta": thr,
            })

            mean_d, max_d, thr = delta_stats(iso, raw, lo, hi)
            rows.append({
                "dataset": prefix,
                "model_family": model_family,
                "range": label,
                "comparison": "isotonic_minus_raw",
                "mean_delta_NB": mean_d,
                "max_delta_NB": max_d,
                "thr_at_max_delta": thr,
            })

    add_block("deep", deep_paths)
    add_block("hybrid", hyb_paths)
    return pd.DataFrame(rows)

def main():
    os.makedirs("results/tables", exist_ok=True)

    # HAM (already present)
    ham_deep = {
        "raw": "results/audit/dca_ham_deep_raw.csv",
        "platt": "results/audit/dca_ham_deep_platt.csv",
        "isotonic": "results/audit/dca_ham_deep_isotonic.csv",
    }
    ham_hyb = {
        "raw": "results/audit/dca_ham_hyb_raw.csv",
        "platt": "results/audit/dca_ham_hyb_platt.csv",
        "isotonic": "results/audit/dca_ham_hyb_isotonic.csv",
    }

    ham = summarize_one("HAM_test", ham_deep, ham_hyb)
    ham_out = "results/tables/table_dca_recalibration_ham.csv"
    ham.to_csv(ham_out, index=False)
    print("Wrote", ham_out)

    # ISIC (only if present; if not, we just skip)
    isic_deep = {
        "raw": "results/audit/dca_isic_deep_raw.csv",
        "platt": "results/audit/dca_isic_deep_platt.csv",
        "isotonic": "results/audit/dca_isic_deep_isotonic.csv",
    }
    isic_hyb = {
        "raw": "results/audit/dca_isic_hyb_raw.csv",
        "platt": "results/audit/dca_isic_hyb_platt.csv",
        "isotonic": "results/audit/dca_isic_hyb_isotonic.csv",
    }

    if all(os.path.exists(p) for p in isic_deep.values()) and all(os.path.exists(p) for p in isic_hyb.values()):
        isic = summarize_one("ISIC_task3", isic_deep, isic_hyb)
        isic_out = "results/tables/table_dca_recalibration_isic.csv"
        isic.to_csv(isic_out, index=False)
        print("Wrote", isic_out)
    else:
        print("ISIC DCA calibrated files not found; skipped ISIC summary.")

if __name__ == "__main__":
    main()
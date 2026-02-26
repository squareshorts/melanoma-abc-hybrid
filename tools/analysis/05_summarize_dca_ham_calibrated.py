import pandas as pd


def summarize_delta(a_path, b_path, name):
    a = pd.read_csv(a_path)
    b = pd.read_csv(b_path)

    m = a.merge(b, on="threshold", suffixes=("_a", "_b"))
    m["delta"] = m["net_benefit_a"] - m["net_benefit_b"]

    mid = m[(m.threshold >= 0.10) & (m.threshold <= 0.90)]

    print("\n" + name)
    print("Mean ΔNB (0.1–0.9):", float(mid["delta"].mean()))
    print("Max  ΔNB:", float(m["delta"].max()))


def main():

    summarize_delta(
        "results/audit/dca_ham_deep_platt.csv",
        "results/audit/dca_ham_deep_raw.csv",
        "DEEP: Platt - Raw"
    )

    summarize_delta(
        "results/audit/dca_ham_deep_isotonic.csv",
        "results/audit/dca_ham_deep_raw.csv",
        "DEEP: Iso - Raw"
    )

    summarize_delta(
        "results/audit/dca_ham_hyb_platt.csv",
        "results/audit/dca_ham_hyb_raw.csv",
        "HYBRID: Platt - Raw"
    )

    summarize_delta(
        "results/audit/dca_ham_hyb_isotonic.csv",
        "results/audit/dca_ham_hyb_raw.csv",
        "HYBRID: Iso - Raw"
    )


if __name__ == "__main__":
    main()
import pandas as pd

def find_max(a_path, b_path, name):
    a = pd.read_csv(a_path)
    b = pd.read_csv(b_path)
    m = a.merge(b, on="threshold", suffixes=("_a","_b"))
    m["delta"] = m["net_benefit_a"] - m["net_benefit_b"]
    j = m["delta"].idxmax()
    r = m.loc[j, ["threshold","net_benefit_a","net_benefit_b","delta"]]
    print("\n" + name)
    print(r.to_string())

def main():
    find_max("results/audit/dca_ham_deep_platt.csv", "results/audit/dca_ham_deep_raw.csv", "DEEP Platt - Raw")
    find_max("results/audit/dca_ham_deep_isotonic.csv", "results/audit/dca_ham_deep_raw.csv", "DEEP Iso - Raw")
    find_max("results/audit/dca_ham_hyb_platt.csv", "results/audit/dca_ham_hyb_raw.csv", "HYB Platt - Raw")
    find_max("results/audit/dca_ham_hyb_isotonic.csv", "results/audit/dca_ham_hyb_raw.csv", "HYB Iso - Raw")

if __name__ == "__main__":
    main()
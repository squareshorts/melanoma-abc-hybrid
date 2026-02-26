import pandas as pd
import numpy as np

d = pd.read_csv("results/audit/dca_ham_deep.csv")
h = pd.read_csv("results/audit/dca_ham_hybrid.csv")

def pick_nb(df):
    cols = [c for c in df.columns]
    # common names
    preferred = ["net_benefit","nb","NetBenefit","netBenefit","model_net_benefit","net_benefit_model"]
    for c in preferred:
        if c in cols: return c
    # heuristic: choose numeric col that isn't threshold and isn't "all"/"none"
    num = [c for c in cols if c != "threshold" and pd.api.types.is_numeric_dtype(df[c])]
    num = [c for c in num if not any(k in c.lower() for k in ["all","none","treat","strategy"])]
    if len(num)==1:
        return num[0]
    raise RuntimeError(f"Could not identify net benefit column. Candidates={num}. All cols={cols}")

nb_d = pick_nb(d)
nb_h = pick_nb(h)

m = d[["threshold", nb_d]].merge(h[["threshold", nb_h]], on="threshold", suffixes=("_deep","_hyb"))
m = m.rename(columns={nb_d+"_deep":"nb_deep", nb_h+"_hyb":"nb_hyb"})
m["delta_nb"] = m["nb_deep"] - m["nb_hyb"]

print("Using columns:", {"deep_nb": nb_d, "hyb_nb": nb_h})
print("Max ΔNB:", float(m["delta_nb"].max()))
mid = m[(m["threshold"]>=0.1) & (m["threshold"]<=0.9)]
print("Mean ΔNB (0.1–0.9):", float(mid["delta_nb"].mean()))
print("ΔNB @ thresholds 0.1,0.3,0.5,0.7,0.9:")
for t in [0.1,0.3,0.5,0.7,0.9]:
    row = m.iloc[(m["threshold"]-t).abs().argsort()[:1]]
    print(f"  t~{t:.1f}  delta_nb={float(row['delta_nb']):.6f}")

m.to_csv("results/audit/dca_ham_delta_deep_minus_hybrid.csv", index=False)
print("Wrote results/audit/dca_ham_delta_deep_minus_hybrid.csv")

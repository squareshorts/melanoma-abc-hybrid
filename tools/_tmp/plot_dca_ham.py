# tools/_tmp/plot_dca_ham.py
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

deep_csv  = Path("results/audit/dca_ham_deep.csv")
hyb_csv   = Path("results/audit/dca_ham_hybrid.csv")
delta_csv = Path("results/audit/dca_ham_delta_deep_minus_hybrid.csv")  # optional

out_dir = Path("results/figures")
out_dir.mkdir(parents=True, exist_ok=True)
out_path = out_dir / "fig_dca_ham.png"

d = pd.read_csv(deep_csv)
h = pd.read_csv(hyb_csv)

# Your tool wrote net benefit for the model under this column name
# (you already detected it as net_benefit_model)
nb_col = "net_benefit_model"
thr_col = "threshold"

if nb_col not in d.columns or nb_col not in h.columns:
    raise RuntimeError(f"Expected column '{nb_col}' in both DCA files.")

plt.figure()
plt.plot(d[thr_col], d[nb_col], label="Deep")
plt.plot(h[thr_col], h[nb_col], label="Hybrid")

# If your DCA CSVs also include treat-all / treat-none baselines, plot them.
# Common names: net_benefit_all, net_benefit_none
for col, lab in [("net_benefit_all", "Treat-all"), ("net_benefit_none", "Treat-none")]:
    if col in d.columns:
        plt.plot(d[thr_col], d[col], label=lab)

# Optional: delta curve (deep - hybrid), if you want it on the same axes.
# If you prefer a separate panel/figure, remove this block.
if delta_csv.exists():
    m = pd.read_csv(delta_csv)
    if "delta_nb" in m.columns and thr_col in m.columns:
        plt.plot(m[thr_col], m["delta_nb"], label="ΔNB (Deep − Hybrid)")

plt.xlabel("Threshold probability")
plt.ylabel("Net benefit")
plt.legend()
plt.tight_layout()
plt.savefig(out_path, dpi=300)
print("Wrote", out_path)
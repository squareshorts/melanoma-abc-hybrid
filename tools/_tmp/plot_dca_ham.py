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

lo, hi = 0.02, 0.50
d = d[(d[thr_col] >= lo) & (d[thr_col] <= hi)].copy()
h = h[(h[thr_col] >= lo) & (h[thr_col] <= hi)].copy()

fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharex=True)
ax = axes[0]
ax.plot(d[thr_col], d[nb_col], label="Deep", linewidth=2)
ax.plot(h[thr_col], h[nb_col], label="Hybrid", linewidth=2)

# If your DCA CSVs also include treat-all / treat-none baselines, plot them.
# Common names: net_benefit_all, net_benefit_none
for col, lab in [("net_benefit_all", "Treat-all"), ("net_benefit_none", "Treat-none")]:
    if col in d.columns:
        ax.plot(d[thr_col], d[col], label=lab, linewidth=1.5, linestyle="--" if lab == "Treat-all" else ":")

# Optional: delta curve (deep - hybrid), if you want it on the same axes.
# If you prefer a separate panel/figure, remove this block.
if delta_csv.exists():
    m = pd.read_csv(delta_csv)
    if "delta_nb" in m.columns and thr_col in m.columns:
        m = m[(m[thr_col] >= lo) & (m[thr_col] <= hi)].copy()
        axes[1].plot(m[thr_col], m["delta_nb"], color="purple", linewidth=2)
        axes[1].axhline(0, color="0.4", linewidth=1, linestyle=":")
        axes[1].set_title("Delta net benefit")
        axes[1].set_ylabel("Deep minus hybrid")

ax.set_title("Net benefit")
ax.set_ylabel("Net benefit")
model_min = min(d[nb_col].min(), h[nb_col].min(), 0.0)
model_max = max(d[nb_col].max(), h[nb_col].max(), 0.0)
ax.set_ylim(model_min - 0.02, model_max + 0.02)
for a in axes:
    a.set_xlabel("Threshold probability")
    a.set_xlim(lo, hi)
ax.legend(frameon=False)
fig.tight_layout()
fig.savefig(out_path, dpi=600)
fig.savefig(out_path.with_suffix(".pdf"))
print("Wrote", out_path)

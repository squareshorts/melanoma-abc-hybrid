import json
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss

RNG = np.random.default_rng(42)
N_BOOT = 5000  # bump if you want tighter CI; 5000 is typically fine

# Inputs (already standardized by your make_audit_preds step)
deep_path   = Path("results/audit/preds_ham_deep.csv")
hybrid_path = Path("results/audit/preds_ham_hybrid.csv")

meta_path   = Path("data/raw/HAM10000/metadata.csv")
split_path  = Path("data/splits/ham_patient_split.json")

out_path = Path("results/audit/paired_cluster_bootstrap_ham_deep_minus_hybrid.csv")

# ---- load preds ----
deep   = pd.read_csv(deep_path)    # columns: image_id,y,p
hybrid = pd.read_csv(hybrid_path)

# Sanity: ensure same images & same labels
m = deep.merge(hybrid, on="image_id", suffixes=("_deep","_hyb"))
if m.empty:
    raise RuntimeError("No overlap between deep and hybrid prediction files by image_id.")

# If y differs due to formatting, force consistency check
y_mismatch = (m["y_deep"].astype(float) != m["y_hyb"].astype(float)).sum()
if y_mismatch:
    raise RuntimeError(f"Label mismatch between deep and hybrid for {y_mismatch} rows.")

m = m.rename(columns={"y_deep":"y", "p_deep":"p_deep", "p_hyb":"p_hyb"})[["image_id","y","p_deep","p_hyb"]]
m["y"] = m["y"].astype(int)

# ---- map image_id -> lesion_id, restricting to TEST ----
meta = pd.read_csv(meta_path)
split = json.load(open(split_path, "r"))

les2split = {}
for k in ["train","val","test"]:
    for lid in split.get(k, []):
        les2split[lid] = k
meta["split"] = meta["lesion_id"].map(les2split)

meta_test = meta[meta["split"] == "test"][["image_id","lesion_id"]].copy()

df = m.merge(meta_test, on="image_id", how="inner")
if df.empty:
    raise RuntimeError("After merging with metadata TEST split, no rows remain. Check image_id formats.")
if df["lesion_id"].isna().any():
    raise RuntimeError("Missing lesion_id after merge; check metadata coverage.")

# clusters
lesions = df["lesion_id"].unique()
n_les = len(lesions)

# ---- bootstrap ----
d_auc = []
d_pr  = []
d_br  = []

degenerate = 0

for _ in range(N_BOOT):
    sampled_lesions = RNG.choice(lesions, size=n_les, replace=True)
    boot = df[df["lesion_id"].isin(sampled_lesions)]

    # Must have both classes in resample for AUC/PR
    if boot["y"].nunique() < 2:
        degenerate += 1
        continue

    y = boot["y"].to_numpy()
    pD = boot["p_deep"].to_numpy()
    pH = boot["p_hyb"].to_numpy()

    aucD = roc_auc_score(y, pD)
    aucH = roc_auc_score(y, pH)
    prD  = average_precision_score(y, pD)
    prH  = average_precision_score(y, pH)
    brD  = brier_score_loss(y, pD)
    brH  = brier_score_loss(y, pH)

    d_auc.append(aucD - aucH)   # deep - hybrid
    d_pr.append(prD - prH)
    d_br.append(brD - brH)      # positive means deep worse (higher Brier)

d_auc = np.array(d_auc, dtype=float)
d_pr  = np.array(d_pr, dtype=float)
d_br  = np.array(d_br, dtype=float)

def summary(x):
    lo, med, hi = np.percentile(x, [2.5, 50, 97.5])
    p_pos = float((x > 0).mean())
    return lo, med, hi, p_pos

auc_lo, auc_med, auc_hi, auc_p = summary(d_auc)
pr_lo,  pr_med,  pr_hi,  pr_p  = summary(d_pr)
br_lo,  br_med,  br_hi,  br_p  = summary(d_br)

out = pd.DataFrame([{
    "N_BOOT_REQUESTED": N_BOOT,
    "N_BOOT_VALID": len(d_auc),
    "N_BOOT_DEGENERATE_SKIPPED": degenerate,
    "N_TEST_IMAGES_USED": len(df),
    "N_TEST_LESIONS_USED": n_les,

    "dAUC_median": auc_med,
    "dAUC_low": auc_lo,
    "dAUC_high": auc_hi,
    "P(dAUC>0)": auc_p,

    "dPR_median": pr_med,
    "dPR_low": pr_lo,
    "dPR_high": pr_hi,
    "P(dPR>0)": pr_p,

    "dBrier_median": br_med,
    "dBrier_low": br_lo,
    "dBrier_high": br_hi,
    "P(dBrier>0)": br_p,
}])

out_path.parent.mkdir(parents=True, exist_ok=True)
out.to_csv(out_path, index=False)

print("Wrote", out_path)
print(out.to_string(index=False))

print("\nInterpretation reminders:")
print("  dAUC = AUC_deep - AUC_hybrid; P(dAUC>0) close to 1 means deep likely better AUC.")
print("  dBrier = Brier_deep - Brier_hybrid; positive means deep is worse calibrated (higher Brier).")

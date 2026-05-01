import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss

# -----------------------------
# Configuration
# -----------------------------
N_BOOT = 5000
RNG = np.random.default_rng(42)

deep_path = Path("results/runs/deep_baseline/isic_task3_predictions_deep.csv")
hyb_path  = Path("results/runs/hybrid/isic_task3_predictions_hybrid.csv")

out_path = Path("results/audit/paired_bootstrap_isic_task3.csv")
out_path.parent.mkdir(parents=True, exist_ok=True)

# -----------------------------
# Load predictions
# -----------------------------
deep = pd.read_csv(deep_path)
hyb  = pd.read_csv(hyb_path)

# Standardize column names
deep = deep.rename(columns={"y_true": "y", "y_prob": "p_deep"})
hyb  = hyb.rename(columns={"y_true": "y", "y_prob": "p_hyb"})

# Merge on image_id
df = deep[["image_id", "y", "p_deep"]].merge(
    hyb[["image_id", "p_hyb"]],
    on="image_id",
    how="inner"
)

df["y"] = df["y"].astype(float)

N = len(df)
# Handle grouping if lesion_id exists
group_col = "lesion_id" if "lesion_id" in df.columns else None
if group_col:
    unique_groups = df[group_col].unique()
    n_groups = len(unique_groups)
    group_to_idx = {g: np.where(df[group_col] == g)[0] for g in unique_groups}

for _ in range(N_BOOT):
    if group_col:
        sampled_groups = RNG.choice(unique_groups, size=n_groups, replace=True)
        sample_idx = np.concatenate([group_to_idx[g] for g in sampled_groups])
    else:
        sample_idx = RNG.choice(indices, size=N, replace=True)
    
    boot = df.iloc[sample_idx]

    if boot["y"].nunique() < 2:
        continue

    y = boot["y"].values

    auc_deep = roc_auc_score(y, boot["p_deep"])
    auc_hyb  = roc_auc_score(y, boot["p_hyb"])

    pr_deep = average_precision_score(y, boot["p_deep"])
    pr_hyb  = average_precision_score(y, boot["p_hyb"])

    br_deep = brier_score_loss(y, boot["p_deep"])
    br_hyb  = brier_score_loss(y, boot["p_hyb"])

    dAUC.append(auc_deep - auc_hyb)
    dPR.append(pr_deep - pr_hyb)
    dBrier.append(br_deep - br_hyb)

def ci(arr):
    return np.percentile(arr, [2.5, 50, 97.5])

auc_ci = ci(dAUC)
pr_ci  = ci(dPR)
br_ci  = ci(dBrier)

result = pd.DataFrame([{
    "N_BOOT_REQUESTED": N_BOOT,
    "N_BOOT_VALID": len(dAUC),
    "N_IMAGES": N,
    "dAUC_median": auc_ci[1],
    "dAUC_low": auc_ci[0],
    "dAUC_high": auc_ci[2],
    "P(dAUC>0)": np.mean(np.array(dAUC) > 0),
    "dPR_median": pr_ci[1],
    "dPR_low": pr_ci[0],
    "dPR_high": pr_ci[2],
    "P(dPR>0)": np.mean(np.array(dPR) > 0),
    "dBrier_median": br_ci[1],
    "dBrier_low": br_ci[0],
    "dBrier_high": br_ci[2],
    "P(dBrier>0)": np.mean(np.array(dBrier) > 0),
}])

result.to_csv(out_path, index=False)

print("Wrote", out_path)
print(result.to_string(index=False))
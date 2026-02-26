import json
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss
from pathlib import Path

N_BOOT = 2000
RNG = np.random.default_rng(42)

pred_files = {
    "deep":   Path("results/audit/preds_ham_deep.csv"),
    "hybrid": Path("results/audit/preds_ham_hybrid.csv"),
}

meta = pd.read_csv("data/raw/HAM10000/metadata.csv")
split = json.load(open("data/splits/ham_patient_split.json"))

# Map lesion -> split
les2split = {}
for k in ["train", "val", "test"]:
    for lid in split.get(k, []):
        les2split[lid] = k

meta["split"] = meta["lesion_id"].map(les2split)

# Keep only test images
meta_test = meta[meta["split"] == "test"][["image_id", "lesion_id"]].copy()

out_rows = []

for tag, path in pred_files.items():
    df = pd.read_csv(path)

    df = df.merge(meta_test, on="image_id", how="inner")

    if df["lesion_id"].isna().any():
        raise RuntimeError("Missing lesion_id after merge")

    lesions = df["lesion_id"].unique()

    aucs = []
    pras = []
    briers = []

    for _ in range(N_BOOT):
        sampled = RNG.choice(lesions, size=len(lesions), replace=True)
        boot = pd.concat([df[df["lesion_id"] == lid] for lid in sampled])

        if boot["y"].nunique() < 2:
            continue

        y = boot["y"].values
        p = boot["p"].values

        aucs.append(roc_auc_score(y, p))
        pras.append(average_precision_score(y, p))
        briers.append(brier_score_loss(y, p))

    def ci(x):
        return np.percentile(x, [2.5, 50, 97.5])

    auc_ci = ci(aucs)
    pr_ci  = ci(pras)
    br_ci  = ci(briers)

    out_rows.append({
        "model": tag,
        "AUC_median": auc_ci[1],
        "AUC_low": auc_ci[0],
        "AUC_high": auc_ci[2],
        "PR_median": pr_ci[1],
        "PR_low": pr_ci[0],
        "PR_high": pr_ci[2],
        "Brier_median": br_ci[1],
        "Brier_low": br_ci[0],
        "Brier_high": br_ci[2],
        "n_boot_valid": len(aucs)
    })

out = pd.DataFrame(out_rows)
out_path = Path("results/audit/cluster_bootstrap_ham_test.csv")
out.to_csv(out_path, index=False)

print("Wrote", out_path)
print(out.to_string(index=False))

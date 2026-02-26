import json
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.linear_model import LogisticRegression

RNG = np.random.default_rng(42)
N_BOOT = 3000

deep_path   = Path("results/audit/preds_ham_deep.csv")
hyb_path    = Path("results/audit/preds_ham_hybrid.csv")
meta_path   = Path("data/raw/HAM10000/metadata.csv")
split_path  = Path("data/splits/ham_patient_split.json")

deep = pd.read_csv(deep_path)
hyb  = pd.read_csv(hyb_path)

m = deep.merge(hyb, on="image_id", suffixes=("_deep","_hyb"))
m = m.rename(columns={"y_deep":"y"})[["image_id","y","p_deep","p_hyb"]]
m["y"] = m["y"].astype(int)

meta = pd.read_csv(meta_path)
split = json.load(open(split_path))

les2split = {}
for k in ["train","val","test"]:
    for lid in split.get(k, []):
        les2split[lid] = k
meta["split"] = meta["lesion_id"].map(les2split)
meta_test = meta[meta["split"]=="test"][["image_id","lesion_id"]]

df = m.merge(meta_test, on="image_id")
lesions = df["lesion_id"].unique()
n_les = len(lesions)

def cal_params(y, p):
    p = np.clip(p, 1e-6, 1-1e-6)
    X = np.log(p/(1-p)).reshape(-1,1)
    model = LogisticRegression(solver="lbfgs")
    model.fit(X, y)
    slope = model.coef_[0][0]
    intercept = model.intercept_[0]
    return intercept, slope

rows = []

for tag in ["deep","hyb"]:
    d_inter = []
    d_slope = []
    for _ in range(N_BOOT):
        sampled = RNG.choice(lesions, size=n_les, replace=True)
        boot = df[df["lesion_id"].isin(sampled)]
        if boot["y"].nunique()<2:
            continue
        y = boot["y"].values
        p = boot[f"p_{tag}"].values
        inter, slope = cal_params(y,p)
        d_inter.append(inter)
        d_slope.append(slope)
    rows.append({
        "model":tag,
        "intercept_median":np.median(d_inter),
        "intercept_low":np.percentile(d_inter,2.5),
        "intercept_high":np.percentile(d_inter,97.5),
        "slope_median":np.median(d_slope),
        "slope_low":np.percentile(d_slope,2.5),
        "slope_high":np.percentile(d_slope,97.5),
    })

out = pd.DataFrame(rows)
out.to_csv("results/audit/calibration_slope_intercept_ham.csv",index=False)
print(out.to_string(index=False))

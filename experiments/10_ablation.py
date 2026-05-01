import os
import numpy as np
import pandas as pd

from src.utils.io import load_config, read_json
from src.data.ham import load_ham_metadata
from src.models.xgb_models import train_xgb, predict_proba
from src.evaluation.metrics import compute_basic
from src.evaluation.bootstrap import bootstrap_ci

def load_embeddings(npy_path, ids_csv):
    E = np.load(npy_path)
    ids = pd.read_csv(ids_csv)["image_id"].tolist()
    return E, ids

def align(df_ids, E, ids):
    idx = {k:i for i,k in enumerate(ids)}
    keep = [idx[i] for i in df_ids["image_id"].tolist() if i in idx]
    df2 = df_ids[df_ids["image_id"].isin(idx.keys())].copy()
    E2 = E[keep]
    df2 = df2.set_index("image_id").loc[[ids[k] for k in keep]].reset_index()
    return df2, E2

if __name__ == "__main__":
    cfg = load_config()
    os.makedirs("results/tables", exist_ok=True)

    split = read_json(cfg["derived"]["split_json"])
    df_meta = load_ham_metadata(cfg["paths"]["ham_metadata"])
    feats = pd.read_csv(cfg["derived"]["ham_abc_csv"])

    train_ids = set(split["train"])
    test_ids = set(split["test"])

    df_train = df_meta[df_meta["lesion_id"].isin(train_ids)][["image_id","label","lesion_id"]].drop_duplicates().merge(feats, on="image_id", how="inner")
    df_test  = df_meta[df_meta["lesion_id"].isin(test_ids)][["image_id","label","lesion_id"]].drop_duplicates().merge(feats, on="image_id", how="inner")

    feat_cols = [c for c in df_train.columns if c not in ["image_id","label","lesion_id"]]
    E_ham, ids_ham = load_embeddings(cfg["derived"]["ham_emb_npy"], cfg["derived"]["ham_emb_ids"])
    df_train_al, Etr = align(df_train[["image_id"]], E_ham, ids_ham)
    df_test_al,  Ete = align(df_test[["image_id"]],  E_ham, ids_ham)

    df_train = df_train.set_index("image_id").loc[df_train_al["image_id"]].reset_index()
    df_test  = df_test.set_index("image_id").loc[df_test_al["image_id"]].reset_index()

    Xtr_abc, ytr = df_train[feat_cols].values, df_train["label"].values
    Xte_abc, yte = df_test[feat_cols].values, df_test["label"].values

    rows = []

    Xtr_full = np.concatenate([Xtr_abc, Etr], axis=1)
    Xte_full = np.concatenate([Xte_abc, Ete], axis=1)
    m_full = train_xgb(Xtr_full, ytr, cfg["xgb"])
    p_full = predict_proba(m_full, Xte_full)
    stats_full = compute_basic(yte, p_full)
    row_full = {"variant":"full_hybrid", **stats_full}
    for k in ["AUC","PR_AUC","Brier"]:
        lo, hi = bootstrap_ci(yte, p_full, k, n=1000, seed=cfg["seed"], thr=0.5, groups=df_test["lesion_id"].values)
        row_full[f"{k}_CI95"] = f"[{lo:.3f}, {hi:.3f}]"
    rows.append(row_full)

    m_noemb = train_xgb(Xtr_abc, ytr, cfg["xgb"])
    p_noemb = predict_proba(m_noemb, Xte_abc)
    stats_noemb = compute_basic(yte, p_noemb)
    row_noemb = {"variant":"no_embeddings", **stats_noemb}
    for k in ["AUC","PR_AUC","Brier"]:
        lo, hi = bootstrap_ci(yte, p_noemb, k, n=1000, seed=cfg["seed"], thr=0.5, groups=df_test["lesion_id"].values)
        row_noemb[f"{k}_CI95"] = f"[{lo:.3f}, {hi:.3f}]"
    rows.append(row_noemb)

    m_noabc = train_xgb(Etr, ytr, cfg["xgb"])
    p_noabc = predict_proba(m_noabc, Ete)
    stats_noabc = compute_basic(yte, p_noabc)
    row_noabc = {"variant":"no_abc", **stats_noabc}
    for k in ["AUC","PR_AUC","Brier"]:
        lo, hi = bootstrap_ci(yte, p_noabc, k, n=1000, seed=cfg["seed"], thr=0.5, groups=df_test["lesion_id"].values)
        row_noabc[f"{k}_CI95"] = f"[{lo:.3f}, {hi:.3f}]"
    rows.append(row_noabc)

    pd.DataFrame(rows).to_csv("results/tables/table_ablation_internal.csv", index=False)
    print(pd.DataFrame(rows))

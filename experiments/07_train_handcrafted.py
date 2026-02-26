import os
import pandas as pd
from src.utils.io import load_config, read_json
from src.data.ham import load_ham_metadata
from src.models.xgb_models import train_xgb, predict_proba
from src.evaluation.metrics import compute_basic
from src.evaluation.bootstrap import bootstrap_ci

if __name__ == "__main__":
    cfg = load_config()
    os.makedirs("results/tables", exist_ok=True)
    os.makedirs("results/runs/handcrafted", exist_ok=True)

    split = read_json(cfg["derived"]["split_json"])
    df_meta = load_ham_metadata(cfg["paths"]["ham_metadata"])
    feats = pd.read_csv(cfg["derived"]["ham_abc_csv"])

    train_ids = set(split["train"])
    test_ids = set(split["test"])

    df_train = df_meta[df_meta["lesion_id"].isin(train_ids)][["image_id","label"]].drop_duplicates().merge(feats, on="image_id", how="inner")
    df_test  = df_meta[df_meta["lesion_id"].isin(test_ids)][["image_id","label"]].drop_duplicates().merge(feats, on="image_id", how="inner")

    feat_cols = [c for c in df_train.columns if c not in ["image_id","label"]]
    Xtr, ytr = df_train[feat_cols].values, df_train["label"].values
    Xte, yte = df_test[feat_cols].values, df_test["label"].values

    model = train_xgb(Xtr, ytr, cfg["xgb"])
    p = predict_proba(model, Xte)

    stats = compute_basic(yte, p, thr=0.5)
    row = {"model":"handcrafted_xgb", **stats}
    for k in ["AUC","PR_AUC","F1","SENS","SPEC","ACC"]:
        lo, hi = bootstrap_ci(yte, p, k, n=1000, seed=cfg["seed"], thr=0.5)
        row[f"{k}_CI95"] = f"[{lo:.3f}, {hi:.3f}]"

    pd.DataFrame([row]).to_csv("results/tables/table_handcrafted_internal.csv", index=False)
    pd.DataFrame({"image_id": df_test["image_id"], "y_true": yte, "y_prob": p}).to_csv(
        "results/runs/handcrafted/ham_test_predictions_handcrafted.csv", index=False
    )
    print(row)

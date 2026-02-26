# tools/shap_group_importance.py
import numpy as np
import pandas as pd
import joblib
import shap

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", required=True)
    ap.add_argument("--X_csv", required=True)   # columns are features
    ap.add_argument("--out_csv", required=True)
    ap.add_argument("--abc_prefixes", nargs="+", default=["A_", "B_", "C_"])
    args = ap.parse_args()

    model = joblib.load(args.model_path)
    X = pd.read_csv(args.X_csv)
    explainer = shap.TreeExplainer(model)
    shap_vals = explainer.shap_values(X)

    abs_mean = np.mean(np.abs(shap_vals), axis=0)
    feat = np.array(X.columns)

    is_abc = np.zeros(len(feat), dtype=bool)
    for p in args.abc_prefixes:
        is_abc |= np.char.startswith(feat.astype(str), p)

    abc_share = abs_mean[is_abc].sum() / abs_mean.sum()
    top_idx = np.argsort(-abs_mean)[:50]
    top_feats = feat[top_idx]

    out = {
        "abc_share_sum_abs_shap": float(abc_share),
        "top50_features": ";".join(top_feats.astype(str))
    }
    pd.DataFrame([out]).to_csv(args.out_csv, index=False)
    print("Wrote", args.out_csv)
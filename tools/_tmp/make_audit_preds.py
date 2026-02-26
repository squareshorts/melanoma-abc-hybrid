import pandas as pd
from pathlib import Path

inputs = {
  "ham_deep":   Path("results/runs/deep_baseline/ham_test_predictions_deep.csv"),
  "ham_hybrid": Path("results/runs/hybrid/ham_test_predictions_hybrid.csv"),
  "isic_deep":  Path("results/runs/deep_baseline/isic_task3_predictions_deep.csv"),
  "isic_hybrid":Path("results/runs/hybrid/isic_task3_predictions_hybrid.csv"),
}

out_dir = Path("results/audit")
out_dir.mkdir(parents=True, exist_ok=True)

for tag, path in inputs.items():
    df = pd.read_csv(path)
    out = df[["image_id","y_true","y_prob"]].copy()
    out.columns = ["image_id","y","p"]
    out_path = out_dir / f"preds_{tag}.csv"
    out.to_csv(out_path, index=False)
    print(f"Wrote {out_path}  rows={len(out)}")

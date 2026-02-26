import json
import pandas as pd

meta = pd.read_csv("data/raw/HAM10000/metadata.csv")
split = json.load(open("data/splits/ham_patient_split.json"))

les2split = {}
for k in ["train","val","test"]:
    for lid in split.get(k, []):
        les2split[lid] = k

meta["split"] = meta["lesion_id"].map(les2split)

print("Split counts (rows):")
print(meta["split"].value_counts(dropna=False).to_string())

dup = meta.dropna(subset=["split"]).groupby("lesion_id")["split"].nunique()
print("\nLesions spanning multiple splits:", int((dup > 1).sum()))

g = meta.dropna(subset=["split"]).groupby("lesion_id")["image_id"].nunique()
print("\nImages per lesion:")
print("  max   =", int(g.max()))
print("  median=", float(g.median()))

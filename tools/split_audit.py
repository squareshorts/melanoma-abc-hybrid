# tools/split_audit.py
import pandas as pd

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--meta_csv", required=True, help="metadata with columns lesion_id, image_id, split")
    ap.add_argument("--lesion_col", default="lesion_id")
    ap.add_argument("--image_col", default="image_id")
    ap.add_argument("--split_col", default="split")
    ap.add_argument("--patient_col", default=None)
    args = ap.parse_args()

    df = pd.read_csv(args.meta_csv)

    for sp in sorted(df[args.split_col].unique()):
        d = df[df[args.split_col] == sp]
        print(sp, "images", d[args.image_col].nunique(), "lesions", d[args.lesion_col].nunique())
        print("max images per lesion:", d.groupby(args.lesion_col)[args.image_col].nunique().max())

    # leakage by lesion_id
    pivot = df.pivot_table(index=args.lesion_col, columns=args.split_col, values=args.image_col, aggfunc="nunique", fill_value=0)
    multi = pivot[(pivot > 0).sum(axis=1) > 1]
    print("lesions spanning multiple splits:", multi.shape[0])

    if args.patient_col and args.patient_col in df.columns:
        pivotp = df.pivot_table(index=args.patient_col, columns=args.split_col, values=args.image_col, aggfunc="nunique", fill_value=0)
        multip = pivotp[(pivotp > 0).sum(axis=1) > 1]
        print("patients spanning multiple splits:", multip.shape[0])
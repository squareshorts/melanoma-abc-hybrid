import os
import cv2
import argparse
from glob import glob
import pandas as pd

from src.utils.io import load_config
from src.segmentation.levelset import segment_levelset
from src.segmentation.baseline import segment_otsu
from src.segmentation.metrics import dice, iou, failure
from src.segmentation.run_batch_parallel import run_segmentation_parallel

def eval_task1(pred_dir, gt_dir, min_area_ratio=0.01, limit=0, start=0):
    gts = sorted(glob(os.path.join(gt_dir, "*.png")))
    start = max(0, int(start))

    if limit and limit > 0:
        gts = gts[start : start + limit]
    else:
        gts = gts[start:]

    rows = []
    for gt_path in gts:
        name = os.path.splitext(os.path.basename(gt_path))[0]
        pred_path = os.path.join(pred_dir, f"{name}.png")
        if not os.path.exists(pred_path):
            continue
        gt = (cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE) > 0).astype("uint8")
        pr = (cv2.imread(pred_path, cv2.IMREAD_GRAYSCALE) > 0).astype("uint8")
        rows.append(
            {
                "image_id": name,
                "dice": dice(pr, gt),
                "iou": iou(pr, gt),
                "failure": int(failure(pr, min_area_ratio=min_area_ratio)),
            }
        )
    df = pd.DataFrame(rows)

    expected = len(gts)
    got = len(df)
    if expected > 0 and got < expected:
        print(
            f"[WARN] Only {got}/{expected} predictions found in {pred_dir} "
            f"for slice start={start}, limit={limit}."
        )

    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ISIC 2018 Task 1 segmentation benchmark")
    parser.add_argument("--limit", type=int, default=0, help="Evaluate only N GT masks (0=all)")
    parser.add_argument("--start", type=int, default=0, help="Start index in GT list (default 0)")
    parser.add_argument("--skip-seg", action="store_true", help="Skip segmentation run and only evaluate")
    parser.add_argument("--only-levelset", action="store_true", help="Run/evaluate only Level Set")
    parser.add_argument("--only-baseline", action="store_true", help="Run/evaluate only Otsu baseline")
    parser.add_argument("--jobs", type=int, default=4, help="Number of parallel workers for segmentation")
    args = parser.parse_args()

    if args.only_levelset and args.only_baseline:
        raise SystemExit("Choose at most one: --only-levelset or --only-baseline")

    cfg = load_config()
    os.makedirs("results/tables", exist_ok=True)

    run_levelset = not args.only_baseline
    run_baseline = not args.only_levelset

    # Segmentation stage (supports start+limit slicing via run_batch.py)
    if not args.skip_seg:
        if run_levelset:
            run_segmentation_parallel(
                cfg["paths"]["isic_task1_images"],
                cfg["derived"]["isic_task1_levelset_dir"],
                lambda rgb: segment_levelset(rgb, **cfg["segmentation"]["levelset"]),
                limit=args.limit,
                start=args.start,
                pattern="*.jpg",
                n_jobs=args.jobs,
            )
        if run_baseline:
            run_segmentation_parallel(
                cfg["paths"]["isic_task1_images"],
                cfg["derived"]["isic_task1_baseline_dir"],
                segment_otsu,
                limit=args.limit,
                start=args.start,
                pattern="*.jpg",
                n_jobs=args.jobs,
            )

    # Evaluation stage (supports start+limit slicing over GT list)
    rows = []
    min_area_ratio = cfg["segmentation"]["postprocess"]["min_area_ratio"]

    if run_levelset:
        df_ls = eval_task1(
            cfg["derived"]["isic_task1_levelset_dir"],
            cfg["paths"]["isic_task1_masks"],
            min_area_ratio=min_area_ratio,
            limit=args.limit,
            start=args.start,
        )
        if df_ls.empty:
            raise SystemExit("LevelSet evaluation empty. Missing predicted masks for this slice.")
        rows.append(
            {
                "method": "LevelSet",
                "dice_mean": float(df_ls["dice"].mean()),
                "iou_mean": float(df_ls["iou"].mean()),
                "failure_rate": float(df_ls["failure"].mean()),
                "N": int(len(df_ls)),
                "start": int(args.start),
                "limit": int(args.limit),
            }
        )

    if run_baseline:
        df_bl = eval_task1(
            cfg["derived"]["isic_task1_baseline_dir"],
            cfg["paths"]["isic_task1_masks"],
            min_area_ratio=min_area_ratio,
            limit=args.limit,
            start=args.start,
        )
        if df_bl.empty:
            raise SystemExit("OtsuBaseline evaluation empty. Missing predicted masks for this slice.")
        rows.append(
            {
                "method": "OtsuBaseline",
                "dice_mean": float(df_bl["dice"].mean()),
                "iou_mean": float(df_bl["iou"].mean()),
                "failure_rate": float(df_bl["failure"].mean()),
                "N": int(len(df_bl)),
                "start": int(args.start),
                "limit": int(args.limit),
            }
        )

    summary = pd.DataFrame(rows)
    summary.to_csv("results/tables/table_II_segmentation_task1.csv", index=False)
    print(summary)
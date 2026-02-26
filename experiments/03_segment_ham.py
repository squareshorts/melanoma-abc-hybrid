import argparse
from src.utils.io import load_config
from src.segmentation.levelset import segment_levelset
from src.segmentation.run_batch_parallel import run_segmentation_parallel

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--jobs", type=int, default=12)
    p.add_argument("--start", type=int, default=0)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--pattern", type=str, default="*.*")
    p.add_argument("--skip-existing", action="store_true")
    args = p.parse_args()

    cfg = load_config()

    run_segmentation_parallel(
        image_dir=cfg["paths"]["ham_images"],
        out_dir=cfg["derived"]["ham_seg_dir"],
        segment_fn=lambda rgb: segment_levelset(rgb, **cfg["segmentation"]["levelset"]),
        start=args.start,
        limit=args.limit,
        pattern=args.pattern,
        n_jobs=args.jobs,
        skip_existing=True if args.skip_existing else True,  # always skip; safe for resume
    )

    print("Wrote masks to:", cfg["derived"]["ham_seg_dir"])
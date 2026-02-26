# scripts/bootstrap_ci.py

import argparse
import glob
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score


DEFAULT_TRUE_CANDS = [
    "y_true", "label", "target", "gt", "is_melanoma", "melanoma", "truth"
]

DEFAULT_SCORE_CANDS = [
    "y_prob",  # primary in your project
    "y_score", "score", "prob", "proba", "p", "p_melanoma", "melanoma_prob",
    "pred_prob", "pred_proba", "prob_malignant", "logit"
]


def find_column(df: pd.DataFrame, candidates):
    cols = list(df.columns)
    lower_map = {c.lower(): c for c in cols}
    for cand in candidates:
        if cand.lower() in lower_map:
            return lower_map[cand.lower()]
    return None


def coerce_binary(y):
    # bool
    if y.dtype == bool:
        return y.astype(int)

    # numeric (0/1 or 0.0/1.0)
    if np.issubdtype(y.dtype, np.number):
        uy = pd.unique(y)
        if set(map(float, uy)).issubset({0.0, 1.0}):
            return y.astype(int)
        raise ValueError(f"y_true appears non-binary numeric: unique={sorted(map(float, uy))[:10]}")

    # string/categorical
    y_str = y.astype(str).str.lower()

    pos_tokens = {"1", "true", "melanoma", "malignant", "pos", "positive"}
    neg_tokens = {"0", "false", "benign", "nevus", "neg", "negative", "non-melanoma", "nonmelanoma"}

    mapped = []
    for v in y_str:
        if v in pos_tokens:
            mapped.append(1)
        elif v in neg_tokens:
            mapped.append(0)
        else:
            raise ValueError(f"Unrecognized y_true token '{v}'. Add mapping in coerce_binary().")

    return np.array(mapped, dtype=int)


def coerce_score(s):
    s = np.asarray(s)
    if not np.issubdtype(s.dtype, np.number):
        raise ValueError("y_score column must be numeric.")
    return s.astype(float)


def bootstrap_ci(y_true, y_score, metric_fn, B=1000, seed=42):
    rng = np.random.default_rng(seed)
    n = len(y_true)

    vals = np.empty(B, dtype=float)

    for b in range(B):
        idx = rng.integers(0, n, size=n)
        yt = y_true[idx]
        ys = y_score[idx]

        # skip degenerate resample
        if len(np.unique(yt)) < 2:
            vals[b] = np.nan
            continue

        vals[b] = metric_fn(yt, ys)

    vals = vals[~np.isnan(vals)]

    mean = float(np.mean(vals))
    lo, hi = np.quantile(vals, [0.025, 0.975]).tolist()

    return float(lo), float(hi), mean


def process_file(csv_path: str, B: int, seed: int):
    df = pd.read_csv(csv_path)

    y_true_col = find_column(df, DEFAULT_TRUE_CANDS)
    y_score_col = find_column(df, DEFAULT_SCORE_CANDS)

    if y_true_col is None or y_score_col is None:
        raise KeyError(
            f"Could not auto-detect columns in {csv_path}.\n"
            f"Columns: {df.columns.tolist()}\n"
            f"Expected y_true like: {DEFAULT_TRUE_CANDS}\n"
            f"Expected y_score like: {DEFAULT_SCORE_CANDS}"
        )

    y_true = coerce_binary(df[y_true_col])
    y_score = coerce_score(df[y_score_col])

    auc_lo, auc_hi, auc_mean = bootstrap_ci(
        y_true, y_score, lambda yt, ys: roc_auc_score(yt, ys), B=B, seed=seed
    )

    ap_lo, ap_hi, ap_mean = bootstrap_ci(
        y_true, y_score, lambda yt, ys: average_precision_score(yt, ys), B=B, seed=seed
    )

    out = {
        "file": csv_path,
        "y_true_col": y_true_col,
        "y_score_col": y_score_col,
        "n": int(len(y_true)),
        "pos": int(np.sum(y_true == 1)),
        "neg": int(np.sum(y_true == 0)),
        "roc_auc_mean": auc_mean,
        "roc_auc_ci_lo": auc_lo,
        "roc_auc_ci_hi": auc_hi,
        "pr_auc_mean": ap_mean,
        "pr_auc_ci_lo": ap_lo,
        "pr_auc_ci_hi": ap_hi,
    }

    name = Path(csv_path).name
    print(f"\n{name}")
    print(f"  n={out['n']}  pos={out['pos']}  neg={out['neg']}")
    print(f"  ROC-AUC: {auc_mean:.4f} [{auc_lo:.4f}, {auc_hi:.4f}]")
    print(f"  PR-AUC:  {ap_mean:.4f} [{ap_lo:.4f}, {ap_hi:.4f}]")

    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--glob", required=True, help="Glob pattern for prediction CSVs")
    ap.add_argument("--B", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="results/bootstrap_ci_summary.csv")
    args = ap.parse_args()

    files = sorted(glob.glob(args.glob, recursive=True))

    if not files:
        print(f"No files matched glob: {args.glob}")
        return

    rows = []
    for fp in files:
        rows.append(process_file(fp, B=args.B, seed=args.seed))

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out_path, index=False)

    print(f"\nWrote summary: {out_path} (n_files={len(rows)})")


if __name__ == "__main__":
    main()
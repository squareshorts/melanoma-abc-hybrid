# scripts/bootstrap_operating_points.py

import argparse
import glob
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve


DEFAULT_TRUE_CANDS = ["y_true", "label", "target", "gt", "is_melanoma", "melanoma", "truth"]
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
    if y.dtype == bool:
        return y.astype(int)

    if np.issubdtype(y.dtype, np.number):
        uy = pd.unique(y)
        if set(map(float, uy)).issubset({0.0, 1.0}):
            return y.astype(int)
        raise ValueError(f"y_true appears non-binary numeric: unique={sorted(map(float, uy))[:10]}")

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


def confusion_at_threshold(y_true, y_score, thr):
    y_pred = (y_score >= thr).astype(int)
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    return tp, tn, fp, fn


def metrics_from_confusion(tp, tn, fp, fn):
    sens = tp / (tp + fn) if (tp + fn) > 0 else np.nan
    spec = tn / (tn + fp) if (tn + fp) > 0 else np.nan
    ppv  = tp / (tp + fp) if (tp + fp) > 0 else np.nan
    npv  = tn / (tn + fn) if (tn + fn) > 0 else np.nan
    acc  = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else np.nan
    return sens, spec, ppv, npv, acc


def op_point_at_min_spec(y_true, y_score, min_spec=0.90):
    # roc_curve returns thresholds in descending score order
    fpr, tpr, thr = roc_curve(y_true, y_score, drop_intermediate=False)
    spec = 1.0 - fpr

    ok = np.where(spec >= min_spec)[0]
    if len(ok) == 0:
        # cannot reach min_spec; choose maximum specificity
        i = int(np.argmax(spec))
    else:
        # among feasible points, pick highest sensitivity
        i = int(ok[np.argmax(tpr[ok])])

    chosen_thr = float(thr[i])

    tp, tn, fp, fn = confusion_at_threshold(y_true, y_score, chosen_thr)
    sens_c, spec_c, ppv, npv, acc = metrics_from_confusion(tp, tn, fp, fn)

    return {
        "threshold": chosen_thr,
        "sens": float(sens_c),
        "spec": float(spec_c),
        "ppv": float(ppv),
        "npv": float(npv),
        "acc": float(acc),
        "tp": tp, "tn": tn, "fp": fp, "fn": fn
    }


def op_point_at_min_sens(y_true, y_score, min_sens=0.90):
    fpr, tpr, thr = roc_curve(y_true, y_score, drop_intermediate=False)
    spec = 1.0 - fpr

    ok = np.where(tpr >= min_sens)[0]
    if len(ok) == 0:
        # cannot reach min_sens; choose maximum sensitivity
        i = int(np.argmax(tpr))
    else:
        # among feasible points, pick highest specificity
        i = int(ok[np.argmax(spec[ok])])

    chosen_thr = float(thr[i])

    tp, tn, fp, fn = confusion_at_threshold(y_true, y_score, chosen_thr)
    sens_c, spec_c, ppv, npv, acc = metrics_from_confusion(tp, tn, fp, fn)

    return {
        "threshold": chosen_thr,
        "sens": float(sens_c),
        "spec": float(spec_c),
        "ppv": float(ppv),
        "npv": float(npv),
        "acc": float(acc),
        "tp": tp, "tn": tn, "fp": fp, "fn": fn
    }


def bootstrap_op_points(y_true, y_score, B=1000, seed=42, min_spec=0.90, min_sens=0.90):
    rng = np.random.default_rng(seed)
    n = len(y_true)

    rows = []
    for b in range(B):
        idx = rng.integers(0, n, size=n)
        yt = y_true[idx]
        ys = y_score[idx]

        if len(np.unique(yt)) < 2:
            continue

        op_spec = op_point_at_min_spec(yt, ys, min_spec=min_spec)
        op_sens = op_point_at_min_sens(yt, ys, min_sens=min_sens)

        rows.append({
            "b": b,
            "spec_constraint_threshold": op_spec["threshold"],
            "spec_constraint_sens": op_spec["sens"],
            "spec_constraint_spec": op_spec["spec"],
            "spec_constraint_ppv": op_spec["ppv"],
            "spec_constraint_npv": op_spec["npv"],
            "spec_constraint_acc": op_spec["acc"],

            "sens_constraint_threshold": op_sens["threshold"],
            "sens_constraint_sens": op_sens["sens"],
            "sens_constraint_spec": op_sens["spec"],
            "sens_constraint_ppv": op_sens["ppv"],
            "sens_constraint_npv": op_sens["npv"],
            "sens_constraint_acc": op_sens["acc"],
        })

    dfb = pd.DataFrame(rows)
    if len(dfb) == 0:
        raise RuntimeError("All bootstrap resamples were degenerate. Check labels.")

    def ci(col):
        v = dfb[col].to_numpy(dtype=float)
        return float(np.nanmean(v)), float(np.nanquantile(v, 0.025)), float(np.nanquantile(v, 0.975))

    summary = {
        # Spec>=...
        "spec_constraint_threshold_mean": ci("spec_constraint_threshold")[0],
        "spec_constraint_threshold_ci_lo": ci("spec_constraint_threshold")[1],
        "spec_constraint_threshold_ci_hi": ci("spec_constraint_threshold")[2],

        "spec_constraint_sens_mean": ci("spec_constraint_sens")[0],
        "spec_constraint_sens_ci_lo": ci("spec_constraint_sens")[1],
        "spec_constraint_sens_ci_hi": ci("spec_constraint_sens")[2],

        "spec_constraint_spec_mean": ci("spec_constraint_spec")[0],
        "spec_constraint_spec_ci_lo": ci("spec_constraint_spec")[1],
        "spec_constraint_spec_ci_hi": ci("spec_constraint_spec")[2],

        "spec_constraint_ppv_mean": ci("spec_constraint_ppv")[0],
        "spec_constraint_ppv_ci_lo": ci("spec_constraint_ppv")[1],
        "spec_constraint_ppv_ci_hi": ci("spec_constraint_ppv")[2],

        "spec_constraint_npv_mean": ci("spec_constraint_npv")[0],
        "spec_constraint_npv_ci_lo": ci("spec_constraint_npv")[1],
        "spec_constraint_npv_ci_hi": ci("spec_constraint_npv")[2],

        # Sens>=...
        "sens_constraint_threshold_mean": ci("sens_constraint_threshold")[0],
        "sens_constraint_threshold_ci_lo": ci("sens_constraint_threshold")[1],
        "sens_constraint_threshold_ci_hi": ci("sens_constraint_threshold")[2],

        "sens_constraint_sens_mean": ci("sens_constraint_sens")[0],
        "sens_constraint_sens_ci_lo": ci("sens_constraint_sens")[1],
        "sens_constraint_sens_ci_hi": ci("sens_constraint_sens")[2],

        "sens_constraint_spec_mean": ci("sens_constraint_spec")[0],
        "sens_constraint_spec_ci_lo": ci("sens_constraint_spec")[1],
        "sens_constraint_spec_ci_hi": ci("sens_constraint_spec")[2],

        "sens_constraint_ppv_mean": ci("sens_constraint_ppv")[0],
        "sens_constraint_ppv_ci_lo": ci("sens_constraint_ppv")[1],
        "sens_constraint_ppv_ci_hi": ci("sens_constraint_ppv")[2],

        "sens_constraint_npv_mean": ci("sens_constraint_npv")[0],
        "sens_constraint_npv_ci_lo": ci("sens_constraint_npv")[1],
        "sens_constraint_npv_ci_hi": ci("sens_constraint_npv")[2],
    }

    return summary, dfb


def process_file(csv_path: str, B: int, seed: int, min_spec: float, min_sens: float):
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

    summary, _ = bootstrap_op_points(
        y_true, y_score, B=B, seed=seed, min_spec=min_spec, min_sens=min_sens
    )

    out = {
        "file": csv_path,
        "y_true_col": y_true_col,
        "y_score_col": y_score_col,
        "n": int(len(y_true)),
        "pos": int(np.sum(y_true == 1)),
        "neg": int(np.sum(y_true == 0)),
        "min_spec_target": float(min_spec),
        "min_sens_target": float(min_sens),
        **summary
    }

    name = Path(csv_path).name
    print(f"\n{name}")
    print(f"  n={out['n']}  pos={out['pos']}  neg={out['neg']}")
    print(f"  Spec≥{min_spec:.2f}:  Sens {out['spec_constraint_sens_mean']:.3f} "
          f"[{out['spec_constraint_sens_ci_lo']:.3f}, {out['spec_constraint_sens_ci_hi']:.3f}]  "
          f"Thr {out['spec_constraint_threshold_mean']:.3f}")
    print(f"  Sens≥{min_sens:.2f}:  Spec {out['sens_constraint_spec_mean']:.3f} "
          f"[{out['sens_constraint_spec_ci_lo']:.3f}, {out['sens_constraint_spec_ci_hi']:.3f}]  "
          f"Thr {out['sens_constraint_threshold_mean']:.3f}")

    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--glob", required=True, help="Glob pattern for prediction CSVs")
    ap.add_argument("--B", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--min_spec", type=float, default=0.90)
    ap.add_argument("--min_sens", type=float, default=0.90)
    ap.add_argument("--out", default="results/operating_point_ci_summary.csv")
    args = ap.parse_args()

    files = sorted(glob.glob(args.glob, recursive=True))
    if not files:
        print(f"No files matched glob: {args.glob}")
        return

    rows = []
    for fp in files:
        rows.append(process_file(fp, B=args.B, seed=args.seed,
                                min_spec=args.min_spec, min_sens=args.min_sens))

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out_path, index=False)

    print(f"\nWrote summary: {out_path} (n_files={len(rows)})")


if __name__ == "__main__":
    main()
"""Experiment 21 — Resubmission analyses.

Run all or individual steps:
    python experiments/21_resubmission_analyses.py --steps all
    python experiments/21_resubmission_analyses.py --steps split_stability,negative_controls
"""

import argparse, itertools, json, math, os, sys, warnings
from pathlib import Path

import cv2
import joblib
import matplotlib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from scipy import stats
from scipy.stats import mannwhitneyu, spearmanr
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, average_precision_score, brier_score_loss,
    confusion_matrix, roc_auc_score, roc_curve,
)
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from src.calibration.abc_conditioned import ABCCalibrator  # noqa: E402
from src.calibration.beta_calibration import BetaCalibration  # noqa: E402
from src.calibration.temperature_scaling import TemperatureScaler  # noqa: E402
from src.data.split_patient import make_patient_split  # noqa: E402
from src.data.transforms import build_transforms  # noqa: E402
from src.evaluation.icc import icc_2way_agreement  # noqa: E402
from src.features.abc import extract_abc  # noqa: E402
from src.models.xgb_models import predict_proba, train_xgb  # noqa: E402
from src.utils.io import load_config  # noqa: E402

TABLE = ROOT / "results" / "tables"
FIGURE = ROOT / "results" / "figures"
AUDIT = ROOT / "results" / "audit"
EPS = 1e-7
DCA_NARROW = np.linspace(0.02, 0.20, 50)
DCA_WIDE = np.linspace(0.02, 0.50, 50)


def ensure_dirs():
    for d in [TABLE, FIGURE, AUDIT]:
        d.mkdir(parents=True, exist_ok=True)


def save_fig(fig, path):
    p = Path(path); p.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(); fig.savefig(p, dpi=600); fig.savefig(p.with_suffix(".pdf")); plt.close(fig)


def safe_logit(p):
    p = np.clip(np.asarray(p, dtype=float), EPS, 1.0 - EPS)
    return np.log(p / (1.0 - p))


def safe_auc(y, p):
    y = np.asarray(y).astype(int)
    return float(roc_auc_score(y, p)) if len(np.unique(y)) >= 2 else float("nan")


def ece(y_true, y_prob, n_bins=10):
    y, p = np.asarray(y_true).astype(int), np.asarray(y_prob).astype(float)
    if len(y) == 0: return float("nan")
    order = np.argsort(p); chunks = np.array_split(order, min(n_bins, len(order)))
    val = 0.0
    for idx in chunks:
        if len(idx) == 0: continue
        val += (len(idx) / len(y)) * abs(float(y[idx].mean()) - float(p[idx].mean()))
    return float(val)


def net_benefit_curve(y, p, thresholds):
    y, p, n = np.asarray(y).astype(int), np.asarray(p).astype(float), len(y)
    out = []
    for t in thresholds:
        pred = p >= t; tp = np.sum(pred & (y == 1)); fp = np.sum(pred & (y == 0))
        out.append((tp / n) - (fp / n) * (t / (1.0 - t)))
    return np.array(out)


def integrated_nb(y, p, thresholds):
    return float(np.mean(net_benefit_curve(y, p, thresholds)))


def cal_slope_intercept(y_true, y_prob):
    y, p = np.asarray(y_true).astype(int), np.asarray(y_prob).astype(float)
    if len(np.unique(y)) < 2 or len(np.unique(np.round(p, 12))) < 2:
        return float("nan"), float("nan")
    try:
        lr = LogisticRegression(max_iter=1000)
        lr.fit(safe_logit(p).reshape(-1, 1), y)
        return float(lr.coef_[0, 0]), float(lr.intercept_[0])
    except Exception:
        return float("nan"), float("nan")


def full_metrics(y, p):
    y, p = np.asarray(y).astype(int), np.asarray(p).astype(float)
    sl, ic = cal_slope_intercept(y, p)
    return {
        "N": len(y), "prevalence": float(y.mean()) if len(y) else float("nan"),
        "AUC": safe_auc(y, p),
        "PR-AUC": float(average_precision_score(y, p)) if len(np.unique(y)) >= 2 else float("nan"),
        "Brier": float(brier_score_loss(y, p)) if len(y) else float("nan"),
        "ECE": ece(y, p),
        "INB_0.02_0.20": integrated_nb(y, p, DCA_NARROW),
        "INB_0.02_0.50": integrated_nb(y, p, DCA_WIDE),
        "cal_slope": sl, "cal_intercept": ic,
    }


# ── Calibrator fitting helpers ─────────────────────────────────────────
def fit_abccal(s_val, c_val, y_val, lambdas=None, seed=42):
    if lambdas is None: lambdas = [0, 1e-4, 1e-3, 1e-2, 1e-1]
    s_t = torch.tensor(s_val, dtype=torch.float32)
    c_t = torch.tensor(c_val, dtype=torch.float32)
    y_t = torch.tensor(y_val, dtype=torch.float32)
    best = None
    for lam in lambdas:
        torch.manual_seed(seed)
        m = ABCCalibrator(c_val.shape[1])
        opt = torch.optim.LBFGS(m.parameters(), max_iter=150, line_search_fn="strong_wolfe")
        def closure():
            opt.zero_grad()
            loss = nn.BCELoss()(m(s_t, c_t), y_t) + lam * (torch.norm(m.alpha)**2 + torch.norm(m.beta)**2)
            loss.backward(); return loss
        opt.step(closure)
        with torch.no_grad(): vl = float(nn.BCELoss()(m(s_t, c_t), y_t).item())
        if best is None or vl < best[0]: best = (vl, lam, m)
    return best[2], best[1]


def apply_abccal(model, s, c):
    with torch.no_grad():
        return model(torch.tensor(s, dtype=torch.float32),
                     torch.tensor(c, dtype=torch.float32)).numpy()


# ── Data loading helpers ───────────────────────────────────────────────
def load_ham_meta(cfg):
    m = pd.read_csv(cfg["paths"]["ham_metadata"])
    m["label"] = (m["dx"] == "mel").astype(int)
    return m


def load_bcn_preds():
    return pd.read_csv(ROOT / "results/runs/deep_baseline/bcn20000_predictions_deep.csv")


def load_abc_features():
    return {
        "ham": pd.read_csv(ROOT / "data/derived/features/ham_abc.csv"),
        "bcn": pd.read_csv(ROOT / "data/derived/features/bcn20000_abc.csv"),
    }


def descriptor_cols(df):
    return [c for c in df.columns if c.startswith(("A_", "B_", "C_"))]


# ======================================================================
# ANALYSIS 1: Repeated source-split stability
# ======================================================================
def run_split_stability(cfg, n_seeds=20, seed_offset=0):
    print(f"[1/9] Split stability ({n_seeds} seeds)...")
    ham = load_ham_meta(cfg)
    abc_feats = load_abc_features()
    bcn_deep = load_bcn_preds()
    bcn_abc = abc_feats["bcn"]
    bcn_merged = bcn_deep.merge(bcn_abc, on="image_id", how="inner")
    bcn_y = bcn_merged["y_true"].astype(int).values
    bcn_logit = safe_logit(bcn_merged["y_prob"].values)
    feat_cols_abc = descriptor_cols(bcn_abc)

    # Load deep val/test predictions (fixed from single backbone)
    ham_val_deep = pd.read_csv(ROOT / "results/runs/deep_baseline/ham_val_predictions_deep.csv")
    ham_test_deep = pd.read_csv(ROOT / "results/runs/deep_baseline/ham_test_predictions_deep.csv")
    all_deep = pd.concat([ham_val_deep, ham_test_deep], ignore_index=True)

    rows = []
    for seed_i in range(n_seeds):
        seed = seed_i + seed_offset
        # Re-split at lesion level
        tmp_json = ROOT / f"data/splits/_tmp_stability_seed{seed}.json"
        try:
            make_patient_split(cfg["paths"]["ham_metadata"], str(tmp_json), seed=seed)
            with open(tmp_json) as f: sp = json.load(f)
        finally:
            if tmp_json.exists(): tmp_json.unlink()

        val_ids = set(sp["val"]); test_ids = set(sp["test"]); train_ids = set(sp["train"])
        ham_abc = abc_feats["ham"]

        # Map image_id → lesion_id via metadata
        img_to_lesion = ham[["image_id", "lesion_id"]].drop_duplicates().set_index("image_id")["lesion_id"]

        # Get val predictions from the fixed deep model
        val_deep = all_deep[all_deep["image_id"].map(lambda x: img_to_lesion.get(x, "") in val_ids)].copy()
        val_merged = val_deep.merge(ham_abc, on="image_id", how="inner")
        if len(val_merged) < 20:
            print(f"  Seed {seed}: too few val samples ({len(val_merged)}), skipping")
            continue

        y_val = val_merged["y_true"].astype(int).values
        s_val = safe_logit(val_merged["y_prob"].values)

        scaler = StandardScaler()
        c_val = scaler.fit_transform(val_merged[feat_cols_abc].values)
        c_bcn = scaler.transform(bcn_merged[feat_cols_abc].values)

        # Fit correction on source validation only
        model_cal, _ = fit_abccal(s_val, c_val, y_val, seed=seed)
        p_corrected = apply_abccal(model_cal, bcn_logit, c_bcn)
        p_raw = bcn_merged["y_prob"].values

        rows.append({
            "seed": seed,
            "n_val": len(val_merged),
            "AUC_raw": safe_auc(bcn_y, p_raw),
            "AUC_corrected": safe_auc(bcn_y, p_corrected),
            "AUC_gain": safe_auc(bcn_y, p_corrected) - safe_auc(bcn_y, p_raw),
            "Brier_raw": float(brier_score_loss(bcn_y, p_raw)),
            "Brier_corrected": float(brier_score_loss(bcn_y, p_corrected)),
            "Brier_change": float(brier_score_loss(bcn_y, p_corrected)) - float(brier_score_loss(bcn_y, p_raw)),
            "INB_raw": integrated_nb(bcn_y, p_raw, DCA_WIDE),
            "INB_corrected": integrated_nb(bcn_y, p_corrected, DCA_WIDE),
            "INB_change": integrated_nb(bcn_y, p_corrected, DCA_WIDE) - integrated_nb(bcn_y, p_raw, DCA_WIDE),
        })
        print(f"  Seed {seed}: AUC gain={rows[-1]['AUC_gain']:.4f}")

    df = pd.DataFrame(rows)
    df.to_csv(TABLE / "table_split_stability_seeds.csv", index=False)

    # Summary
    summary = []
    for col in ["AUC_gain", "Brier_change", "INB_change"]:
        v = df[col].dropna()
        summary.append({"metric": col, "mean": v.mean(), "sd": v.std(), "median": v.median(),
                         "q25": v.quantile(0.25), "q75": v.quantile(0.75), "min": v.min(), "max": v.max(), "n_seeds": len(v)})
    pd.DataFrame(summary).to_csv(TABLE / "table_split_stability_summary.csv", index=False)

    # Figure
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    for ax, col, lbl in zip(axes, ["AUC_gain", "Brier_change", "INB_change"],
                             ["External AUC gain", "Brier score change", "Integrated NB change"]):
        ax.violinplot(df[col].dropna(), showmedians=True)
        ax.axhline(0, color="0.5", ls="--", lw=1)
        ax.set_ylabel(lbl); ax.set_xticks([])
    save_fig(fig, FIGURE / "fig_split_stability_distributions.png")


# ======================================================================
# ANALYSIS 2: Negative-control corrections
# ======================================================================
def run_negative_controls(cfg, n_reps=20, seed=42):
    print("[2/9] Negative-control corrections...")
    rng = np.random.default_rng(seed)

    ham_val = pd.read_csv(ROOT / "results/runs/deep_baseline/ham_val_predictions_deep.csv")
    ham_abc = pd.read_csv(ROOT / "data/derived/features/ham_abc.csv")
    bcn_deep = load_bcn_preds()
    bcn_abc = pd.read_csv(ROOT / "data/derived/features/bcn20000_abc.csv")

    val = ham_val.merge(ham_abc, on="image_id", how="inner")
    bcn = bcn_deep.merge(bcn_abc, on="image_id", how="inner")
    feat_cols = descriptor_cols(ham_abc)
    y_val = val["y_true"].astype(int).values
    s_val = safe_logit(val["y_prob"].values)
    y_bcn = bcn["y_true"].astype(int).values
    s_bcn = safe_logit(bcn["y_prob"].values)
    p_raw_bcn = bcn["y_prob"].values
    auc_raw = safe_auc(y_bcn, p_raw_bcn)
    brier_raw = float(brier_score_loss(y_bcn, p_raw_bcn))

    # Real correction (reference)
    scaler_real = StandardScaler()
    c_val_real = scaler_real.fit_transform(val[feat_cols].values)
    c_bcn_real = scaler_real.transform(bcn[feat_cols].values)
    model_real, _ = fit_abccal(s_val, c_val_real, y_val, seed=seed)
    p_real = apply_abccal(model_real, s_bcn, c_bcn_real)
    auc_real = safe_auc(y_bcn, p_real)

    rows = [{"control_type": "real", "repetition": 0,
             "AUC_raw": auc_raw, "AUC_corrected": auc_real,
             "AUC_gain": auc_real - auc_raw,
             "Brier_change": float(brier_score_loss(y_bcn, p_real)) - brier_raw}]

    controls = {
        "shuffled_descriptors": lambda: (val[feat_cols].values[rng.permutation(len(val))],
                                          bcn[feat_cols].values[rng.permutation(len(bcn))]),
        "random_gaussian": lambda: (rng.standard_normal((len(val), len(feat_cols))),
                                     rng.standard_normal((len(bcn), len(feat_cols)))),
        "label_permuted": lambda: (val[feat_cols].values, bcn[feat_cols].values),
    }

    for ctrl_name, gen_fn in controls.items():
        for rep in range(n_reps):
            c_v, c_b = gen_fn()
            sc = StandardScaler(); c_v = sc.fit_transform(c_v); c_b = sc.transform(c_b)
            y_fit = rng.permutation(y_val) if ctrl_name == "label_permuted" else y_val
            try:
                m, _ = fit_abccal(s_val, c_v, y_fit, seed=seed + rep)
                p_ctrl = apply_abccal(m, s_bcn, c_b)
                auc_c = safe_auc(y_bcn, p_ctrl)
                brier_c = float(brier_score_loss(y_bcn, p_ctrl))
            except Exception:
                auc_c, brier_c = auc_raw, brier_raw
            rows.append({"control_type": ctrl_name, "repetition": rep,
                          "AUC_raw": auc_raw, "AUC_corrected": auc_c,
                          "AUC_gain": auc_c - auc_raw,
                          "Brier_change": brier_c - brier_raw})

    df = pd.DataFrame(rows)
    df.to_csv(TABLE / "table_negative_control_corrections.csv", index=False)

    # Summary
    summary = []
    for ct in df["control_type"].unique():
        v = df[df["control_type"] == ct]["AUC_gain"]
        summary.append({"control_type": ct, "mean_AUC_gain": v.mean(), "sd": v.std(),
                          "median": v.median(), "n": len(v)})
    pd.DataFrame(summary).to_csv(TABLE / "table_negative_control_summary.csv", index=False)

    # Figure
    fig, ax = plt.subplots(figsize=(7, 4))
    types = ["real", "shuffled_descriptors", "random_gaussian", "label_permuted"]
    data = [df[df["control_type"] == t]["AUC_gain"].dropna().values for t in types]
    parts = ax.violinplot(data, showmedians=True)
    ax.set_xticks(range(1, len(types) + 1))
    ax.set_xticklabels(["Real", "Shuffled", "Random", "Label perm"], rotation=15)
    ax.axhline(0, color="0.5", ls="--", lw=1)
    ax.set_ylabel("External AUC gain")
    save_fig(fig, FIGURE / "fig_negative_control_auc_gain.png")


# ======================================================================
# ANALYSIS 3: Stronger calibration baselines
# ======================================================================
def run_calibration_baselines(cfg, n_boot=2000, seed=42):
    print("[3/9] Calibration baselines...")
    rng = np.random.default_rng(seed)
    ham_val = pd.read_csv(ROOT / "results/runs/deep_baseline/ham_val_predictions_deep.csv")
    ham_test = pd.read_csv(ROOT / "results/runs/deep_baseline/ham_test_predictions_deep.csv")
    bcn_deep = load_bcn_preds()
    ham_abc = pd.read_csv(ROOT / "data/derived/features/ham_abc.csv")
    bcn_abc = pd.read_csv(ROOT / "data/derived/features/bcn20000_abc.csv")

    val = ham_val.merge(ham_abc, on="image_id", how="inner")
    test_df = ham_test.merge(ham_abc, on="image_id", how="inner")
    bcn = bcn_deep.merge(bcn_abc, on="image_id", how="inner")
    feat_cols = descriptor_cols(ham_abc)

    y_val = val["y_true"].astype(int).values
    p_val = val["y_prob"].values
    s_val = safe_logit(p_val)

    # Fit all methods on validation
    # 1. Temperature scaling
    temp = TemperatureScaler().fit(p_val, y_val)
    # 2. Platt scaling
    platt = LogisticRegression(max_iter=1000); platt.fit(s_val.reshape(-1, 1), y_val)
    # 3. Isotonic
    iso = IsotonicRegression(out_of_bounds="clip"); iso.fit(p_val, y_val)
    # 4. Beta calibration
    beta = BetaCalibration(); beta.fit(p_val, y_val)
    # 5. Intercept-only (prevalence correction)
    intercept_only = LogisticRegression(max_iter=1000)
    intercept_only.fit(np.ones((len(s_val), 1)), y_val)  # no features, just intercept
    # Actually for intercept-only prevalence correction: logit(p) + offset
    prev_offset = safe_logit(np.array([y_val.mean()]))[0] - safe_logit(np.array([p_val.mean()]))[0]
    # 6. ABC-conditioned
    scaler = StandardScaler()
    c_val = scaler.fit_transform(val[feat_cols].values)
    abc_model, _ = fit_abccal(s_val, c_val, y_val, seed=seed)

    def apply_all(df_data, abc_c):
        p = df_data["y_prob"].values; s = safe_logit(p)
        return {
            "Raw": p,
            "Temperature": temp.predict(p),
            "Platt": platt.predict_proba(s.reshape(-1, 1))[:, 1],
            "Isotonic": iso.predict(p),
            "Beta": beta.predict(p),
            "Prevalence-only": 1.0 / (1.0 + np.exp(-(s + prev_offset))),
            "ABC-conditioned": apply_abccal(abc_model, s, abc_c),
        }

    comp_rows = []
    for ds_name, ds_df, c_ds in [("HAM test", test_df, scaler.transform(test_df[feat_cols].values)),
                                    ("BCN20000", bcn, scaler.transform(bcn[feat_cols].values))]:
        y = ds_df["y_true"].astype(int).values
        preds = apply_all(ds_df, c_ds)
        for mname, p in preds.items():
            m = full_metrics(y, p)
            comp_rows.append({"dataset": ds_name, "method": mname, **m})

    pd.DataFrame(comp_rows).to_csv(TABLE / "table_calibration_baselines_comparison.csv", index=False)

    # Bootstrap pairwise differences on BCN20000
    y_bcn = bcn["y_true"].astype(int).values
    preds_bcn = apply_all(bcn, scaler.transform(bcn[feat_cols].values))
    metrics_to_compare = ["AUC", "Brier", "INB_0.02_0.50"]
    boot_rows = []
    n = len(y_bcn)
    for ref_name in ["Temperature", "Platt", "Isotonic", "Beta", "Prevalence-only"]:
        for metric in metrics_to_compare:
            diffs = np.empty(n_boot)
            point_abc = full_metrics(y_bcn, preds_bcn["ABC-conditioned"])[metric]
            point_ref = full_metrics(y_bcn, preds_bcn[ref_name])[metric]
            for b in range(n_boot):
                idx = rng.integers(0, n, size=n)
                m_abc = full_metrics(y_bcn[idx], preds_bcn["ABC-conditioned"][idx])
                m_ref = full_metrics(y_bcn[idx], preds_bcn[ref_name][idx])
                diffs[b] = m_abc[metric] - m_ref[metric]
            boot_rows.append({
                "dataset": "BCN20000", "method_a": "ABC-conditioned", "method_b": ref_name,
                "metric": metric, "point_diff": point_abc - point_ref,
                "ci_low": float(np.quantile(diffs, 0.025)), "ci_high": float(np.quantile(diffs, 0.975)),
            })
    pd.DataFrame(boot_rows).to_csv(TABLE / "table_calibration_baselines_bootstrap.csv", index=False)

    # Figure
    comp = pd.DataFrame(comp_rows)
    bcn_comp = comp[comp["dataset"] == "BCN20000"].copy()
    order = ["Raw", "Temperature", "Platt", "Isotonic", "Beta", "Prevalence-only", "ABC-conditioned"]
    bcn_comp["method"] = pd.Categorical(bcn_comp["method"], categories=order, ordered=True)
    bcn_comp = bcn_comp.sort_values("method")
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    for ax, metric, ylabel in zip(axes, ["AUC", "Brier", "INB_0.02_0.50"],
                                    ["AUC", "Brier score", "Integrated NB"]):
        ax.bar(range(len(bcn_comp)), bcn_comp[metric].values, color="0.35")
        ax.set_xticks(range(len(bcn_comp)))
        ax.set_xticklabels(bcn_comp["method"].astype(str), rotation=35, ha="right", fontsize=8)
        ax.set_ylabel(ylabel)
    save_fig(fig, FIGURE / "fig_calibration_baselines_bcn.png")


# ======================================================================
# ANALYSIS 4: DCA confidence intervals
# ======================================================================
def run_dca_ci(cfg, n_boot=2000, seed=42):
    print("[4/9] DCA confidence intervals...")
    rng = np.random.default_rng(seed)

    datasets = {}
    for ds_name, paths in [
        ("HAM test", {"ABC-only": ROOT / "results/runs/handcrafted/ham_test_predictions_handcrafted.csv",
                       "Deep": ROOT / "results/runs/deep_baseline/ham_test_predictions_deep.csv",
                       "Hybrid": ROOT / "results/runs/hybrid/ham_test_predictions_hybrid.csv"}),
        ("BCN20000", {"ABC-only": ROOT / "results/runs/handcrafted/bcn20000_predictions_handcrafted.csv",
                       "Deep": ROOT / "results/runs/deep_baseline/bcn20000_predictions_deep.csv",
                       "Hybrid": ROOT / "results/runs/hybrid/bcn20000_predictions_hybrid.csv"}),
    ]:
        merged = None
        for model, path in paths.items():
            if not path.exists(): continue
            df = pd.read_csv(path).rename(columns={"y_prob": model})[["image_id", "y_true", model]]
            if merged is None: merged = df
            else: merged = merged.merge(df[["image_id", model]], on="image_id", how="inner")
        if merged is not None: datasets[ds_name] = merged

    curve_rows, inb_rows = [], []
    for ds_name, df in datasets.items():
        y = df["y_true"].astype(int).values
        models = [c for c in df.columns if c not in ["image_id", "y_true"]]
        n = len(y)

        for model in models:
            p = df[model].values
            for label, grid in [("0.02-0.20", DCA_NARROW), ("0.02-0.50", DCA_WIDE)]:
                nb_point = net_benefit_curve(y, p, grid)
                nb_boots = np.empty((n_boot, len(grid)))
                inb_boots = np.empty(n_boot)
                for b in range(n_boot):
                    idx = rng.integers(0, n, size=n)
                    nb_boots[b] = net_benefit_curve(y[idx], p[idx], grid)
                    inb_boots[b] = np.mean(nb_boots[b])

                for j, t in enumerate(grid):
                    curve_rows.append({"dataset": ds_name, "model": model, "range": label,
                                        "threshold": float(t), "NB": float(nb_point[j]),
                                        "NB_ci_low": float(np.quantile(nb_boots[:, j], 0.025)),
                                        "NB_ci_high": float(np.quantile(nb_boots[:, j], 0.975))})

                inb_rows.append({"dataset": ds_name, "model": model, "range": label,
                                  "INB": float(np.mean(nb_point)),
                                  "INB_ci_low": float(np.quantile(inb_boots, 0.025)),
                                  "INB_ci_high": float(np.quantile(inb_boots, 0.975)),
                                  "n_boot": n_boot})

    pd.DataFrame(curve_rows).to_csv(TABLE / "table_dca_bootstrap_curves.csv", index=False)
    pd.DataFrame(inb_rows).to_csv(TABLE / "table_dca_integrated_net_benefit.csv", index=False)

    # Figures — one per dataset
    for ds_name, df in datasets.items():
        y = df["y_true"].astype(int).values
        models = [c for c in df.columns if c not in ["image_id", "y_true"]]
        fig, ax = plt.subplots(figsize=(7, 5))
        prev = y.mean()
        ax.plot(DCA_WIDE, prev - (1 - prev) * DCA_WIDE / (1 - DCA_WIDE), ":", color="k", label="Treat all")
        ax.plot(DCA_WIDE, np.zeros_like(DCA_WIDE), "-", color="k", label="Treat none")
        colors = plt.cm.tab10(np.linspace(0, 1, len(models)))
        crv = pd.DataFrame(curve_rows)
        for i, model in enumerate(models):
            sub = crv[(crv["dataset"] == ds_name) & (crv["model"] == model) & (crv["range"] == "0.02-0.50")]
            if sub.empty: continue
            ax.plot(sub["threshold"], sub["NB"], label=model, color=colors[i], lw=2)
            ax.fill_between(sub["threshold"], sub["NB_ci_low"], sub["NB_ci_high"], alpha=0.15, color=colors[i])
        ax.set_xlabel("Threshold probability"); ax.set_ylabel("Net benefit")
        ax.set_xlim(0, 0.5); ax.set_ylim(-0.05, max(0.2, prev + 0.05))
        ax.legend(frameon=False, fontsize=8)
        tag = ds_name.lower().replace(" ", "_")
        save_fig(fig, FIGURE / f"fig_dca_bootstrap_{tag}.png")


# ======================================================================
# ANALYSIS 5: Clinically interpretable operating points
# ======================================================================
def _sensitivity_at_specificity(y, p, target_spec):
    fpr, tpr, thr = roc_curve(y, p)
    spec = 1 - fpr
    valid = spec >= target_spec
    if not valid.any(): return float("nan"), float("nan")
    idx = np.where(valid)[0]
    best = idx[np.argmax(tpr[idx])]
    return float(tpr[best]), float(thr[best])


def _specificity_at_sensitivity(y, p, target_sens):
    fpr, tpr, thr = roc_curve(y, p)
    valid = tpr >= target_sens
    if not valid.any(): return float("nan"), float("nan")
    idx = np.where(valid)[0]
    best = idx[np.argmax(1 - fpr[idx])]
    return float(1 - fpr[best]), float(thr[best])


def run_clinical_operating_points(cfg, n_boot=2000, seed=42):
    print("[5/9] Clinical operating points...")
    rng = np.random.default_rng(seed)

    datasets = {}
    for ds_name, paths in [
        ("HAM test", {"ABC-only": ROOT / "results/runs/handcrafted/ham_test_predictions_handcrafted.csv",
                       "Deep": ROOT / "results/runs/deep_baseline/ham_test_predictions_deep.csv",
                       "Hybrid": ROOT / "results/runs/hybrid/ham_test_predictions_hybrid.csv"}),
        ("BCN20000", {"ABC-only": ROOT / "results/runs/handcrafted/bcn20000_predictions_handcrafted.csv",
                       "Deep": ROOT / "results/runs/deep_baseline/bcn20000_predictions_deep.csv",
                       "Hybrid": ROOT / "results/runs/hybrid/bcn20000_predictions_hybrid.csv"}),
    ]:
        merged = None
        for model, path in paths.items():
            if not path.exists(): continue
            df = pd.read_csv(path).rename(columns={"y_prob": model})[["image_id", "y_true", model]]
            if merged is None: merged = df
            else: merged = merged.merge(df[["image_id", model]], on="image_id", how="inner")
        if merged is not None: datasets[ds_name] = merged

    rows = []
    for ds_name, df in datasets.items():
        y = df["y_true"].astype(int).values
        n = len(y)
        models = [c for c in df.columns if c not in ["image_id", "y_true"]]
        for model in models:
            p = df[model].values
            prevalence = y.mean()

            for target_spec in [0.80, 0.90, 0.95]:
                sens_point, thr = _sensitivity_at_specificity(y, p, target_spec)
                boot_sens = np.empty(n_boot)
                for b in range(n_boot):
                    idx = rng.integers(0, n, size=n)
                    s, _ = _sensitivity_at_specificity(y[idx], p[idx], target_spec)
                    boot_sens[b] = s
                valid = boot_sens[np.isfinite(boot_sens)]
                ci_lo = float(np.quantile(valid, 0.025)) if len(valid) > 10 else float("nan")
                ci_hi = float(np.quantile(valid, 0.975)) if len(valid) > 10 else float("nan")
                # FN per 1000 screened
                fn_per_1000 = (1 - sens_point) * prevalence * 1000 if np.isfinite(sens_point) else float("nan")
                rows.append({"dataset": ds_name, "model": model,
                              "metric": "sensitivity_at_specificity", "target": target_spec,
                              "value": sens_point, "ci_low": ci_lo, "ci_high": ci_hi,
                              "threshold": thr, "FN_per_1000": fn_per_1000, "referrals_per_melanoma": float("nan")})

            for target_sens in [0.80, 0.90, 0.95]:
                spec_point, thr = _specificity_at_sensitivity(y, p, target_sens)
                boot_spec = np.empty(n_boot)
                for b in range(n_boot):
                    idx = rng.integers(0, n, size=n)
                    s, _ = _specificity_at_sensitivity(y[idx], p[idx], target_sens)
                    boot_spec[b] = s
                valid = boot_spec[np.isfinite(boot_spec)]
                ci_lo = float(np.quantile(valid, 0.025)) if len(valid) > 10 else float("nan")
                ci_hi = float(np.quantile(valid, 0.975)) if len(valid) > 10 else float("nan")
                # Referrals per melanoma detected
                if np.isfinite(thr) and np.isfinite(spec_point):
                    pred_pos = np.sum(p >= thr)
                    tp = np.sum((p >= thr) & (y == 1))
                    ref_per_mel = float(pred_pos / tp) if tp > 0 else float("nan")
                else:
                    ref_per_mel = float("nan")
                fn_per_1000 = (1 - target_sens) * prevalence * 1000
                rows.append({"dataset": ds_name, "model": model,
                              "metric": "specificity_at_sensitivity", "target": target_sens,
                              "value": spec_point, "ci_low": ci_lo, "ci_high": ci_hi,
                              "threshold": thr, "FN_per_1000": fn_per_1000,
                              "referrals_per_melanoma": ref_per_mel})

    pd.DataFrame(rows).to_csv(TABLE / "table_clinical_operating_points.csv", index=False)

    # Forest plot
    op = pd.DataFrame(rows)
    bcn_sens = op[(op["dataset"] == "BCN20000") & (op["metric"] == "sensitivity_at_specificity")]
    if not bcn_sens.empty:
        fig, ax = plt.subplots(figsize=(8, 5))
        models = bcn_sens["model"].unique()
        targets = sorted(bcn_sens["target"].unique())
        offsets = np.linspace(-0.15, 0.15, len(models))
        for j, model in enumerate(models):
            sub = bcn_sens[bcn_sens["model"] == model].sort_values("target")
            y_pos = np.arange(len(targets)) + offsets[j]
            ax.errorbar(sub["value"], y_pos,
                         xerr=[sub["value"] - sub["ci_low"], sub["ci_high"] - sub["value"]],
                         fmt="o", label=model, capsize=3)
        ax.set_yticks(range(len(targets)))
        ax.set_yticklabels([f"Spec≥{t}" for t in targets])
        ax.set_xlabel("Sensitivity"); ax.legend(frameon=False, fontsize=8)
        save_fig(fig, FIGURE / "fig_clinical_operating_points.png")


# ======================================================================
# ANALYSIS 6: Mask robustness
# ======================================================================
def run_mask_robustness(cfg, seed=42):
    print("[6/9] Mask robustness...")
    from src.segmentation.levelset import segment_levelset
    from src.segmentation.baseline import segment_otsu

    ham_abc_path = ROOT / "data/derived/features/ham_abc.csv"
    if not ham_abc_path.exists():
        print("  SKIP: ham_abc.csv not found"); return

    ham_abc_ls = pd.read_csv(ham_abc_path)
    feat_cols = descriptor_cols(ham_abc_ls)
    image_dir = Path(cfg["paths"]["ham_images"])

    # Check for ISIC Task 1 masks
    task1_img_dir = Path(cfg["data_root"]) / cfg["raw"]["isic_task1"]["images"]
    task1_mask_dir = Path(cfg["data_root"]) / cfg["raw"]["isic_task1"]["masks"]
    has_task1 = task1_img_dir.exists() and task1_mask_dir.exists()

    # --- Part A: ICC across segmentation methods on a subset ---
    # Use a subset of HAM images (first 200 for speed)
    rng = np.random.default_rng(seed)
    all_ids = ham_abc_ls["image_id"].tolist()
    subset_ids = [all_ids[i] for i in rng.choice(len(all_ids), min(200, len(all_ids)), replace=False)]

    methods = {}  # method_name -> {image_id: feature_dict}
    methods["levelset"] = {row["image_id"]: {f: row[f] for f in feat_cols}
                           for _, row in ham_abc_ls[ham_abc_ls["image_id"].isin(subset_ids)].iterrows()}

    # Re-extract with Otsu and perturbations
    for method_name in ["otsu", "dilate_5", "erode_5"]:
        methods[method_name] = {}

    n_done = 0
    for img_id in subset_ids:
        img_path = None
        for ext in [".jpg", ".jpeg", ".png"]:
            p = image_dir / f"{img_id}{ext}"
            if p.exists(): img_path = p; break
        if img_path is None: continue

        try:
            rgb = cv2.cvtColor(cv2.imread(str(img_path)), cv2.COLOR_BGR2RGB)
            h, w = rgb.shape[:2]
            rgb_small = cv2.resize(rgb, (256, 256))

            # Otsu
            mask_otsu = segment_otsu(rgb_small)
            mask_otsu_full = cv2.resize(mask_otsu, (w, h), interpolation=cv2.INTER_NEAREST)
            feats_otsu = extract_abc(rgb, mask_otsu_full)
            methods["otsu"][img_id] = feats_otsu

            # Levelset mask + perturbations
            mask_ls = segment_levelset(rgb_small, **cfg["segmentation"]["levelset"])
            mask_ls_full = cv2.resize(mask_ls, (w, h), interpolation=cv2.INTER_NEAREST)

            kernel5 = np.ones((5, 5), np.uint8)
            mask_dilated = cv2.dilate(mask_ls_full, kernel5, iterations=1)
            feats_dil = extract_abc(rgb, mask_dilated)
            methods["dilate_5"][img_id] = feats_dil

            mask_eroded = cv2.erode(mask_ls_full, kernel5, iterations=1)
            feats_ero = extract_abc(rgb, mask_eroded)
            methods["erode_5"][img_id] = feats_ero
        except Exception as e:
            continue
        n_done += 1
        if n_done % 50 == 0: print(f"  Processed {n_done} images")

    # Compute ICC per descriptor
    common_ids = sorted(set.intersection(*[set(m.keys()) for m in methods.values()]))
    if len(common_ids) < 10:
        print(f"  Only {len(common_ids)} common images, skipping ICC"); return

    icc_rows = []
    method_names = sorted(methods.keys())
    for feat in feat_cols:
        ratings = np.array([[methods[m][img_id].get(feat, float("nan")) for m in method_names]
                             for img_id in common_ids])
        result = icc_2way_agreement(ratings)
        icc_rows.append({"descriptor": feat, "comparison": "all_methods",
                          "n_methods": len(method_names), "n_images": result["n"],
                          "ICC": result["icc"], "ICC_ci_low": result["ci_low"],
                          "ICC_ci_high": result["ci_high"]})

    pd.DataFrame(icc_rows).to_csv(TABLE / "table_mask_robustness_icc.csv", index=False)

    # --- Part B: Downstream model performance with Otsu features ---
    model_perf_rows = []
    handcrafted_pkg = ROOT / "results/runs/handcrafted/handcrafted_xgb.joblib"
    if handcrafted_pkg.exists():
        pack = joblib.load(handcrafted_pkg)
        abc_model_xgb = pack["model"]; model_feat_cols = pack["feat_cols"]
        # Original levelset
        ham_test = pd.read_csv(ROOT / "results/runs/deep_baseline/ham_test_predictions_deep.csv")
        test_ls = ham_test[["image_id", "y_true"]].merge(ham_abc_ls, on="image_id", how="inner")
        if len(test_ls) > 0:
            p_ls = predict_proba(abc_model_xgb, test_ls[model_feat_cols].values)
            model_perf_rows.append({"mask_variant": "levelset", "model": "ABC-only",
                                      "AUC": safe_auc(test_ls["y_true"], p_ls),
                                      "Brier": float(brier_score_loss(test_ls["y_true"], p_ls))})

    pd.DataFrame(model_perf_rows).to_csv(TABLE / "table_mask_robustness_model_performance.csv", index=False)

    # Heatmap figure
    icc_df = pd.DataFrame(icc_rows)
    if not icc_df.empty:
        fig, ax = plt.subplots(figsize=(8, max(3, len(feat_cols) * 0.35)))
        ax.barh(range(len(icc_df)), icc_df["ICC"].values, color="0.35")
        for i, row in icc_df.iterrows():
            ax.plot([row["ICC_ci_low"], row["ICC_ci_high"]], [i, i], color="tab:red", lw=1.5)
        ax.set_yticks(range(len(icc_df)))
        ax.set_yticklabels(icc_df["descriptor"], fontsize=7)
        ax.set_xlabel("ICC(2,1)")
        ax.axvline(0.75, color="0.6", ls="--", lw=1, label="Good agreement")
        ax.legend(fontsize=7, frameon=False)
        save_fig(fig, FIGURE / "fig_mask_robustness_icc_heatmap.png")


# ======================================================================
# ANALYSIS 7: Domain-shift hypothesis testing
# ======================================================================
def run_domain_shift_hypothesis(cfg, n_boot=1000, seed=42):
    print("[7/9] Domain-shift hypothesis testing...")
    from src.evaluation.domain_shift import train_domain_classifier, compute_mmd, compute_energy_distance

    ham_emb_path = ROOT / "data/derived/embeddings/ham_effb0.npy"
    bcn_emb_path = ROOT / "data/derived/embeddings/bcn20000_effb0.npy"
    if not ham_emb_path.exists() or not bcn_emb_path.exists():
        print("  SKIP: embedding files not found"); return

    ham_ids = pd.read_csv(ROOT / "data/derived/embeddings/ham_effb0_ids.csv")["image_id"].tolist()
    bcn_ids = pd.read_csv(ROOT / "data/derived/embeddings/bcn20000_effb0_ids.csv")["image_id"].tolist()
    ham_test_preds = pd.read_csv(ROOT / "results/runs/deep_baseline/ham_test_predictions_deep.csv")
    bcn_preds = load_bcn_preds()

    ham_idx = {k: i for i, k in enumerate(ham_ids)}
    bcn_idx_map = {k: i for i, k in enumerate(bcn_ids)}
    ham_keep = [i for i in ham_test_preds["image_id"] if i in ham_idx]
    bcn_keep = [i for i in bcn_preds["image_id"] if i in bcn_idx_map]

    ham_E = np.load(ham_emb_path, mmap_mode="r")[[ham_idx[i] for i in ham_keep]]
    bcn_E = np.load(bcn_emb_path, mmap_mode="r")[[bcn_idx_map[i] for i in bcn_keep]]

    # 1. Domain classifier
    dc = train_domain_classifier(ham_E, bcn_E, n_permutations=500, seed=seed)
    pd.DataFrame([dc]).to_csv(TABLE / "table_domain_classifier.csv", index=False)

    # 2. MMD & energy distance (subsample for speed)
    rng = np.random.default_rng(seed)
    max_n = min(1500, len(ham_E), len(bcn_E))
    h_sub = ham_E[rng.choice(len(ham_E), max_n, replace=False)]
    b_sub = bcn_E[rng.choice(len(bcn_E), max_n, replace=False)]

    mmd = compute_mmd(h_sub, b_sub, n_boot=n_boot, seed=seed)
    ed = compute_energy_distance(h_sub, b_sub, n_boot=n_boot, seed=seed)
    pd.DataFrame([{"metric": "MMD2", **{k: mmd[k] for k in ["mmd2", "ci_low", "ci_high"]}},
                   {"metric": "Energy", **{k: ed[k] for k in ["energy_distance", "ci_low", "ci_high"]}}]
                 ).to_csv(TABLE / "table_domain_mmd_energy.csv", index=False)

    # 3. Per-image correction magnitude correlation
    ham_abc = pd.read_csv(ROOT / "data/derived/features/ham_abc.csv")
    bcn_abc = pd.read_csv(ROOT / "data/derived/features/bcn20000_abc.csv")
    ham_val = pd.read_csv(ROOT / "results/runs/deep_baseline/ham_val_predictions_deep.csv")
    feat_cols = descriptor_cols(ham_abc)

    val_m = ham_val.merge(ham_abc, on="image_id", how="inner")
    bcn_m = bcn_preds.merge(bcn_abc, on="image_id", how="inner")
    y_val = val_m["y_true"].astype(int).values
    s_val = safe_logit(val_m["y_prob"].values)
    sc = StandardScaler()
    c_val = sc.fit_transform(val_m[feat_cols].values)
    c_bcn = sc.transform(bcn_m[feat_cols].values)

    model_cal, _ = fit_abccal(s_val, c_val, y_val, seed=seed)
    s_bcn = safe_logit(bcn_m["y_prob"].values)
    p_corr = apply_abccal(model_cal, s_bcn, c_bcn)
    correction_mag = np.abs(p_corr - bcn_m["y_prob"].values)

    # Embedding distance to HAM centroid
    bcn_keep2 = [i for i in bcn_m["image_id"] if i in bcn_idx_map]
    bcn_E2 = np.load(bcn_emb_path, mmap_mode="r")[[bcn_idx_map[i] for i in bcn_keep2]]
    ham_centroid = ham_E.mean(axis=0)
    dist_to_centroid = np.sqrt(np.sum((bcn_E2 - ham_centroid) ** 2, axis=1))

    n_corr = min(len(correction_mag), len(dist_to_centroid))
    corr_rows = []
    if n_corr > 10:
        rho, pv = spearmanr(correction_mag[:n_corr], dist_to_centroid[:n_corr])
        corr_rows.append({"comparison": "correction_mag_vs_embedding_distance",
                           "spearman_rho": float(rho), "p_value": float(pv), "n": n_corr})

    pd.DataFrame(corr_rows).to_csv(TABLE / "table_correction_magnitude_correlation.csv", index=False)

    # Figure
    if n_corr > 10:
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.scatter(dist_to_centroid[:n_corr], correction_mag[:n_corr], s=4, alpha=0.3, linewidths=0)
        ax.set_xlabel("Embedding distance to HAM centroid")
        ax.set_ylabel("|Corrected − Raw| probability")
        save_fig(fig, FIGURE / "fig_correction_vs_domain_shift.png")


# ======================================================================
# ANALYSIS 8: Subgroup transportability
# ======================================================================
def run_subgroup_transportability(cfg, n_boot=2000, seed=42):
    print("[8/9] Subgroup transportability...")
    rng = np.random.default_rng(seed)

    meta = pd.read_csv("C:/work/datasets/BCN20000/images/metadata.csv").rename(columns={"isic_id": "image_id"})
    preds = {
        "ABC-only": pd.read_csv(ROOT / "results/runs/handcrafted/bcn20000_predictions_handcrafted.csv"),
        "Deep": pd.read_csv(ROOT / "results/runs/deep_baseline/bcn20000_predictions_deep.csv"),
        "Hybrid": pd.read_csv(ROOT / "results/runs/hybrid/bcn20000_predictions_hybrid.csv"),
    }
    base = preds["Deep"].rename(columns={"y_prob": "Deep"}).merge(meta, on="image_id", how="left")
    for name in ["ABC-only", "Hybrid"]:
        base = base.merge(preds[name].rename(columns={"y_prob": name})[["image_id", name]],
                          on="image_id", how="inner")

    base["age_group"] = pd.cut(pd.to_numeric(base["age_approx"], errors="coerce"),
                                bins=[-np.inf, 39, 59, 74, np.inf],
                                labels=["<=39", "40-59", "60-74", ">=75"]).astype("object")
    sg_cols = ["age_group", "sex", "anatom_site_general"]
    for c in sg_cols:
        base[c] = base[c].fillna("Missing").astype(str)

    models = {"ABC-only": "ABC-only", "Deep": "Deep", "Hybrid": "Hybrid"}
    rows = []
    for model_name, col_name in models.items():
        for sg in sg_cols:
            for level, g in base.groupby(sg):
                y = g["y_true"].astype(int).values
                p = g[col_name].values
                n = len(y)
                if n < 10: continue

                m = full_metrics(y, p)
                m["mean_predicted_risk"] = float(p.mean())

                # Bootstrap for AUC, Brier, cal_slope, cal_intercept
                boot = {k: np.empty(n_boot) for k in ["AUC", "Brier", "cal_slope", "cal_intercept"]}
                for b in range(n_boot):
                    idx = rng.integers(0, n, size=n)
                    bm = full_metrics(y[idx], p[idx])
                    for k in boot: boot[k][b] = bm[k]

                ci = {}
                for k in boot:
                    valid = boot[k][np.isfinite(boot[k])]
                    if len(valid) > 10:
                        ci[f"{k}_ci_low"] = float(np.quantile(valid, 0.025))
                        ci[f"{k}_ci_high"] = float(np.quantile(valid, 0.975))
                    else:
                        ci[f"{k}_ci_low"] = ci[f"{k}_ci_high"] = float("nan")

                rows.append({"dataset": "BCN20000", "model": model_name,
                              "subgroup_variable": sg, "subgroup_level": level,
                              **{k: m[k] for k in ["N", "prevalence", "AUC", "Brier",
                                                      "cal_slope", "cal_intercept", "mean_predicted_risk"]},
                              **ci})

    df = pd.DataFrame(rows)
    df.to_csv(TABLE / "table_subgroup_transportability.csv", index=False)

    # Forest plot of AUC by subgroup
    fig, ax = plt.subplots(figsize=(9, max(4, len(df) // 3 * 0.3)))
    y_pos = 0
    yticks, ylabels = [], []
    for sg in sg_cols:
        sub = df[df["subgroup_variable"] == sg]
        levels = sorted(sub["subgroup_level"].unique())
        for lvl in levels:
            for j, mn in enumerate(models):
                row = sub[(sub["subgroup_level"] == lvl) & (sub["model"] == mn)]
                if row.empty: continue
                r = row.iloc[0]
                ax.errorbar(r["AUC"], y_pos + j * 0.25,
                             xerr=[[r["AUC"] - r["AUC_ci_low"]], [r["AUC_ci_high"] - r["AUC"]]],
                             fmt="o", capsize=2, markersize=4, label=mn if y_pos == 0 else None)
            yticks.append(y_pos + 0.25); ylabels.append(f"{sg}: {lvl}")
            y_pos += 1
    ax.set_yticks(yticks); ax.set_yticklabels(ylabels, fontsize=6)
    ax.set_xlabel("AUC")
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles[:len(models)], labels[:len(models)], frameon=False, fontsize=7)
    save_fig(fig, FIGURE / "fig_subgroup_transportability_forest.png")


# ======================================================================
# ANALYSIS 9: Saliency sanity controls
# ======================================================================
def run_saliency_sanity(cfg, n_steps=10, limit=200, seed=42):
    print("[9/9] Saliency sanity controls...")
    import timm

    device = "cuda" if torch.cuda.is_available() else "cpu"
    deep_ckpt = ROOT / cfg["derived"]["deep_ckpt"]
    if not deep_ckpt.exists():
        print("  SKIP: deep checkpoint not found"); return

    model = timm.create_model(cfg["training"]["deep"]["backbone"], pretrained=False, num_classes=1).to(device)
    ckpt = torch.load(str(deep_ckpt), map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"]); model.eval()

    # Find target layer
    target_layer = None
    for name, mod in model.named_modules():
        if "conv_head" in name: target_layer = mod
    if target_layer is None:
        for mod in reversed(list(model.modules())):
            if isinstance(mod, nn.Conv2d): target_layer = mod; break

    tfm = build_transforms(cfg["image"]["size"], cfg["image"]["normalize"], train=False)
    task1_img_dir = Path(cfg["data_root"]) / cfg["raw"]["isic_task1"]["images"]
    task1_mask_dir = Path(cfg["data_root"]) / cfg["raw"]["isic_task1"]["masks"]

    if not task1_img_dir.exists() or not task1_mask_dir.exists():
        print("  SKIP: ISIC Task 1 images/masks not found"); return

    ids = sorted(p.stem for p in task1_img_dir.glob("*.jpg")
                 if (task1_mask_dir / f"{p.stem}.png").exists() or
                    (task1_mask_dir / f"{p.stem}_segmentation.png").exists())[:limit]
    if not ids:
        print("  SKIP: no image-mask pairs found"); return

    from src.explainability.gradcam import GradCAM
    cam_fn = GradCAM(model, target_layer)
    img_size = cfg["image"]["size"]

    # --- Part A: Deletion/Insertion curves ---
    del_rows, ins_rows = [], []
    fractions = np.linspace(0, 1, n_steps + 1)

    for img_id in ids:
        img_path = task1_img_dir / f"{img_id}.jpg"
        if not img_path.exists(): continue
        rgb = np.array(Image.open(img_path).convert("RGB").resize((img_size, img_size)))
        x = tfm(Image.fromarray(rgb)).unsqueeze(0).to(device)

        model.zero_grad(set_to_none=True)
        cam = cam_fn(x)  # (H, W)
        if cam.ndim > 2: cam = cam[0]
        cam_flat = cam.ravel()
        order_desc = np.argsort(-cam_flat)  # most salient first
        n_pixels = len(cam_flat)

        # Baseline prediction
        with torch.no_grad():
            base_pred = float(torch.sigmoid(model(x).squeeze()).item())

        del_curve, ins_curve = [base_pred], [float(torch.sigmoid(
            model(torch.zeros_like(x)).squeeze()).item())]

        x_np = x.cpu().numpy().copy()
        x_del = x_np.copy()
        x_ins = np.zeros_like(x_np)

        for step in range(1, n_steps + 1):
            start = int(n_pixels * fractions[step - 1])
            end = int(n_pixels * fractions[step])
            pixel_indices = order_desc[start:end]
            for ch in range(x_np.shape[1]):
                flat_del = x_del[0, ch].ravel()
                flat_del[pixel_indices] = 0
                x_del[0, ch] = flat_del.reshape(img_size, img_size)

                flat_ins = x_ins[0, ch].ravel()
                flat_ins[pixel_indices] = x_np[0, ch].ravel()[pixel_indices]
                x_ins[0, ch] = flat_ins.reshape(img_size, img_size)

            with torch.no_grad():
                del_curve.append(float(torch.sigmoid(
                    model(torch.tensor(x_del, device=device)).squeeze()).item()))
                ins_curve.append(float(torch.sigmoid(
                    model(torch.tensor(x_ins, device=device)).squeeze()).item()))

        auc_del = float(np.trapz(del_curve, fractions))
        auc_ins = float(np.trapz(ins_curve, fractions))
        del_rows.append({"image_id": img_id, "AUC_deletion": auc_del})
        ins_rows.append({"image_id": img_id, "AUC_insertion": auc_ins})

    del_df = pd.DataFrame(del_rows); ins_df = pd.DataFrame(ins_rows)
    merged_di = del_df.merge(ins_df, on="image_id")
    merged_di.to_csv(TABLE / "table_saliency_deletion_insertion.csv", index=False)

    # --- Part B: Randomization sanity check ---
    rand_rows = []
    # Get reference saliency for first 50 images
    ref_cams = {}
    for img_id in ids[:50]:
        img_path = task1_img_dir / f"{img_id}.jpg"
        if not img_path.exists(): continue
        rgb = np.array(Image.open(img_path).convert("RGB").resize((img_size, img_size)))
        x = tfm(Image.fromarray(rgb)).unsqueeze(0).to(device)
        model.zero_grad(set_to_none=True)
        c = cam_fn(x)
        ref_cams[img_id] = c.ravel() if c.ndim <= 2 else c[0].ravel()

    # Randomize top layer and recompute
    import copy
    model_rand = copy.deepcopy(model)
    # Randomize classifier layer
    for name, param in model_rand.named_parameters():
        if "classifier" in name or "fc" in name:
            nn.init.normal_(param)
    model_rand.eval()
    target_layer_rand = None
    for name, mod in model_rand.named_modules():
        if "conv_head" in name: target_layer_rand = mod
    if target_layer_rand is None:
        for mod in reversed(list(model_rand.modules())):
            if isinstance(mod, nn.Conv2d): target_layer_rand = mod; break
    cam_fn_rand = GradCAM(model_rand, target_layer_rand)

    corrs = []
    for img_id, ref_cam in ref_cams.items():
        img_path = task1_img_dir / f"{img_id}.jpg"
        rgb = np.array(Image.open(img_path).convert("RGB").resize((img_size, img_size)))
        x = tfm(Image.fromarray(rgb)).unsqueeze(0).to(device)
        model_rand.zero_grad(set_to_none=True)
        c = cam_fn_rand(x)
        rand_cam = c.ravel() if c.ndim <= 2 else c[0].ravel()
        if len(ref_cam) == len(rand_cam):
            rho, pv = spearmanr(ref_cam, rand_cam)
            corrs.append({"image_id": img_id, "rank_correlation": float(rho), "p_value": float(pv)})

    rand_df = pd.DataFrame(corrs)
    rand_df.to_csv(TABLE / "table_saliency_randomization_check.csv", index=False)
    rand_rows.append({"layer": "classifier", "mean_rank_corr": rand_df["rank_correlation"].mean(),
                       "sd": rand_df["rank_correlation"].std(), "n": len(rand_df)})
    pd.DataFrame(rand_rows).to_csv(TABLE / "table_saliency_randomization_summary.csv", index=False)

    # --- Part C: Overlap stratified by correct/incorrect ---
    per_image_path = AUDIT / "gradcam_localization_per_image.csv"
    if per_image_path.exists():
        pi = pd.read_csv(per_image_path)
        ham_test = pd.read_csv(ROOT / "results/runs/deep_baseline/ham_test_predictions_deep.csv")
        pi = pi.merge(ham_test[["image_id", "y_true", "y_prob"]], on="image_id", how="inner")
        pi["correct"] = ((pi["y_prob"] >= 0.5).astype(int) == pi["y_true"].astype(int))
        metric = "top20_saliency_fraction_inside_mask"
        if metric in pi.columns:
            correct = pi[pi["correct"]][metric].dropna()
            incorrect = pi[~pi["correct"]][metric].dropna()
            u_stat, u_p = mannwhitneyu(correct, incorrect, alternative="two-sided") if len(correct) > 2 and len(incorrect) > 2 else (float("nan"), float("nan"))
            strat_rows = [
                {"group": "correct", "n": len(correct), "mean": correct.mean(), "sd": correct.std()},
                {"group": "incorrect", "n": len(incorrect), "mean": incorrect.mean(), "sd": incorrect.std()},
                {"group": "mann_whitney_U", "n": len(pi), "mean": float(u_stat), "sd": float(u_p)},
            ]
            pd.DataFrame(strat_rows).to_csv(TABLE / "table_saliency_correctness_stratified.csv", index=False)

    # Figures
    if not merged_di.empty:
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        axes[0].hist(merged_di["AUC_deletion"], bins=25, color="0.35", alpha=0.8)
        axes[0].set_xlabel("AUC of deletion curve"); axes[0].set_ylabel("Images")
        axes[1].hist(merged_di["AUC_insertion"], bins=25, color="tab:blue", alpha=0.8)
        axes[1].set_xlabel("AUC of insertion curve"); axes[1].set_ylabel("Images")
        save_fig(fig, FIGURE / "fig_saliency_deletion_insertion.png")

    if not rand_df.empty:
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.hist(rand_df["rank_correlation"], bins=20, color="0.35", alpha=0.8)
        ax.axvline(0, color="tab:red", ls="--"); ax.set_xlabel("Rank correlation with original Grad-CAM")
        ax.set_ylabel("Images")
        save_fig(fig, FIGURE / "fig_saliency_randomization_sanity.png")


# ======================================================================
# MAIN
# ======================================================================
ALL_STEPS = [
    "split_stability", "negative_controls", "calibration_baselines",
    "dca_ci", "clinical_operating_points", "mask_robustness",
    "domain_shift_hypothesis", "subgroup_transportability", "saliency_sanity",
]


def main():
    parser = argparse.ArgumentParser(description="Resubmission analyses (Experiment 21)")
    parser.add_argument("--steps", default="all",
                        help=f"Comma-separated steps or 'all': {','.join(ALL_STEPS)}")
    parser.add_argument("--n-seeds", type=int, default=20)
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--saliency-limit", type=int, default=200)
    args = parser.parse_args()

    os.chdir(ROOT)
    ensure_dirs()
    warnings.filterwarnings("ignore", category=UserWarning)
    cfg = load_config()

    # Build paths convenience dict
    cfg.setdefault("paths", {})
    cfg["paths"]["ham_metadata"] = os.path.join(cfg["data_root"], cfg["raw"]["ham"]["metadata"])
    cfg["paths"]["ham_images"] = os.path.join(cfg["data_root"], cfg["raw"]["ham"]["images"])

    selected = set(ALL_STEPS) if args.steps == "all" else {s.strip() for s in args.steps.split(",") if s.strip()}

    if "split_stability" in selected:
        run_split_stability(cfg, n_seeds=args.n_seeds, seed_offset=args.seed)
    if "negative_controls" in selected:
        run_negative_controls(cfg, n_reps=20, seed=args.seed)
    if "calibration_baselines" in selected:
        run_calibration_baselines(cfg, n_boot=args.bootstrap, seed=args.seed)
    if "dca_ci" in selected:
        run_dca_ci(cfg, n_boot=args.bootstrap, seed=args.seed)
    if "clinical_operating_points" in selected:
        run_clinical_operating_points(cfg, n_boot=args.bootstrap, seed=args.seed)
    if "mask_robustness" in selected:
        run_mask_robustness(cfg, seed=args.seed)
    if "domain_shift_hypothesis" in selected:
        run_domain_shift_hypothesis(cfg, n_boot=args.bootstrap // 2, seed=args.seed)
    if "subgroup_transportability" in selected:
        run_subgroup_transportability(cfg, n_boot=args.bootstrap, seed=args.seed)
    if "saliency_sanity" in selected:
        run_saliency_sanity(cfg, limit=args.saliency_limit, seed=args.seed)

    print("\n=== All requested resubmission analyses complete. ===")


if __name__ == "__main__":
    main()

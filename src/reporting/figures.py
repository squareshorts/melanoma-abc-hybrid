import os
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, precision_recall_curve
from src.evaluation.calibration import calibration_stats

def _ensure(p):
    os.makedirs(os.path.dirname(p), exist_ok=True)

def plot_roc(y_true, y_score, out_path, title="ROC"):
    _ensure(out_path)
    fpr, tpr, _ = roc_curve(y_true, y_score)
    plt.figure()
    plt.plot(fpr, tpr)
    plt.plot([0,1],[0,1], linestyle="--")
    plt.xlabel("FPR"); plt.ylabel("TPR"); plt.title(title)
    plt.tight_layout(); plt.savefig(out_path, dpi=600); plt.close()

def plot_pr(y_true, y_score, out_path, title="PR"):
    _ensure(out_path)
    prec, rec, _ = precision_recall_curve(y_true, y_score)
    plt.figure()
    plt.plot(rec, prec)
    plt.xlabel("Recall"); plt.ylabel("Precision"); plt.title(title)
    plt.tight_layout(); plt.savefig(out_path, dpi=600); plt.close()

def plot_calibration(y_true, y_prob, out_path, title="Calibration"):
    _ensure(out_path)
    mean_pred, frac_pos, brier = calibration_stats(y_true, y_prob, n_bins=10)
    plt.figure()
    plt.plot(mean_pred, frac_pos, marker="o")
    plt.plot([0,1],[0,1], linestyle="--")
    plt.xlabel("Mean predicted probability")
    plt.ylabel("Fraction of positives")
    plt.title(f"{title} (Brier={brier:.3f})")
    plt.tight_layout(); plt.savefig(out_path, dpi=600); plt.close()

def _save_with_vector(fig, out_path):
    _ensure(out_path)
    fig.tight_layout()
    fig.savefig(out_path, dpi=600)
    root, _ = os.path.splitext(out_path)
    fig.savefig(root + ".pdf")
    plt.close(fig)

def plot_roc_multi(series, out_path, title="ROC"):
    fig, ax = plt.subplots(figsize=(6, 5))
    for label, y_true, y_score in series:
        fpr, tpr, _ = roc_curve(y_true, y_score)
        ax.plot(fpr, tpr, label=label, linewidth=2)
    ax.plot([0, 1], [0, 1], linestyle="--", color="0.4", linewidth=1)
    ax.set_xlabel("False-positive rate")
    ax.set_ylabel("True-positive rate")
    ax.set_title(title)
    ax.legend(frameon=False)
    _save_with_vector(fig, out_path)

def plot_pr_multi(series, out_path, title="Precision-recall"):
    fig, ax = plt.subplots(figsize=(6, 5))
    for label, y_true, y_score in series:
        precision, recall, _ = precision_recall_curve(y_true, y_score)
        ax.plot(recall, precision, label=label, linewidth=2)
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title(title)
    ax.legend(frameon=False)
    _save_with_vector(fig, out_path)

def plot_calibration_multi(series, out_path, title="Calibration"):
    fig, ax = plt.subplots(figsize=(6, 5))
    for label, y_true, y_prob in series:
        mean_pred, frac_pos, brier = calibration_stats(y_true, y_prob, n_bins=10)
        ax.plot(mean_pred, frac_pos, marker="o", label=f"{label} (Brier={brier:.3f})", linewidth=2)
    ax.plot([0, 1], [0, 1], linestyle="--", color="0.4", linewidth=1)
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Fraction of positives")
    ax.set_title(title)
    ax.legend(frameon=False, fontsize=9)
    _save_with_vector(fig, out_path)

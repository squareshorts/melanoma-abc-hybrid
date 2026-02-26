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
    plt.tight_layout(); plt.savefig(out_path, dpi=200); plt.close()

def plot_pr(y_true, y_score, out_path, title="PR"):
    _ensure(out_path)
    prec, rec, _ = precision_recall_curve(y_true, y_score)
    plt.figure()
    plt.plot(rec, prec)
    plt.xlabel("Recall"); plt.ylabel("Precision"); plt.title(title)
    plt.tight_layout(); plt.savefig(out_path, dpi=200); plt.close()

def plot_calibration(y_true, y_prob, out_path, title="Calibration"):
    _ensure(out_path)
    mean_pred, frac_pos, brier = calibration_stats(y_true, y_prob, n_bins=10)
    plt.figure()
    plt.plot(mean_pred, frac_pos, marker="o")
    plt.plot([0,1],[0,1], linestyle="--")
    plt.xlabel("Mean predicted probability")
    plt.ylabel("Fraction of positives")
    plt.title(f"{title} (Brier={brier:.3f})")
    plt.tight_layout(); plt.savefig(out_path, dpi=200); plt.close()

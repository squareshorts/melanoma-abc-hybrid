import numpy as np
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    brier_score_loss,
    f1_score,
    accuracy_score,
    confusion_matrix,
)

def safe_auc(y_true, y_score):
    y_true = np.asarray(y_true)
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return roc_auc_score(y_true, y_score)

def compute_basic(y_true, y_score, thr=0.5):
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score).astype(float)
    y_pred = (y_score >= thr).astype(int)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0,1]).ravel()
    sens = tp / (tp + fn + 1e-8)
    spec = tn / (tn + fp + 1e-8)

    return {
        "AUC": safe_auc(y_true, y_score),
        "PR_AUC": average_precision_score(y_true, y_score) if len(np.unique(y_true))>1 else float("nan"),
        "Brier": brier_score_loss(y_true, y_score),
        "F1": f1_score(y_true, y_pred, zero_division=0),
        "ACC": accuracy_score(y_true, y_pred),
        "SENS": sens,
        "SPEC": spec,
        "TP": int(tp), "TN": int(tn), "FP": int(fp), "FN": int(fn),
        "THR": float(thr),
    }

def threshold_at_specificity(y_true, y_score, target_spec=0.90):
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score).astype(float)
    thresholds = np.unique(np.round(y_score, 6))
    best = None
    for t in thresholds:
        y_pred = (y_score >= t).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0,1]).ravel()
        spec = tn / (tn + fp + 1e-8)
        sens = tp / (tp + fn + 1e-8)
        if spec >= target_spec:
            if best is None or sens > best[0]:
                best = (sens, float(t), float(spec))
    return 1.0 if best is None else float(best[1])

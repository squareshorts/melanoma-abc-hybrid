import numpy as np
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss

def calibration_stats(y_true, y_prob, n_bins=10):
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)
    frac_pos, mean_pred = calibration_curve(y_true, y_prob, n_bins=n_bins, strategy="uniform")
    brier = brier_score_loss(y_true, y_prob)
    return mean_pred, frac_pos, float(brier)

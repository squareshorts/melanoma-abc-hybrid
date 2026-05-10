"""Temperature scaling — single-parameter post-hoc calibration."""

import numpy as np
import torch
import torch.nn as nn

EPS = 1e-7


def _safe_logit_np(p):
    p = np.clip(np.asarray(p, dtype=float), EPS, 1.0 - EPS)
    return np.log(p / (1.0 - p))


class TemperatureScaler:
    """Learn a single temperature T on validation logits via NLL.

    After fitting:  calibrated_prob = sigmoid(logit / T)
    """

    def __init__(self):
        self.temperature = 1.0

    def fit(self, y_prob, y_true, max_iter=100):
        """Fit temperature on validation probabilities and labels."""
        s = torch.tensor(_safe_logit_np(y_prob), dtype=torch.float32)
        y = torch.tensor(np.asarray(y_true, dtype=float), dtype=torch.float32)

        log_T = nn.Parameter(torch.tensor(0.0))  # T = exp(log_T) to keep T > 0
        opt = torch.optim.LBFGS([log_T], max_iter=max_iter, line_search_fn="strong_wolfe")

        def closure():
            opt.zero_grad()
            T = torch.exp(log_T)
            p = torch.sigmoid(s / T)
            loss = nn.BCELoss()(p, y)
            loss.backward()
            return loss

        opt.step(closure)
        self.temperature = float(torch.exp(log_T).item())
        return self

    def predict(self, y_prob):
        """Apply temperature scaling to probabilities."""
        s = _safe_logit_np(y_prob)
        calibrated_logit = s / self.temperature
        return 1.0 / (1.0 + np.exp(-calibrated_logit))

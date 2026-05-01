import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from scipy import stats

# ---- Fast DeLong implementation ----

def compute_midrank(x):
    J = np.argsort(x)
    Z = x[J]
    N = len(x)
    T = np.zeros(N)
    i = 0
    while i < N:
        j = i
        while j < N and Z[j] == Z[i]:
            j += 1
        T[i:j] = 0.5*(i + j - 1) + 1
        i = j
    T2 = np.empty(N)
    T2[J] = T
    return T2

def delong_components(predictions, labels):
    m = np.sum(labels == 1)
    n = np.sum(labels == 0)
    pos_preds = predictions[labels == 1]
    neg_preds = predictions[labels == 0]
    tx = compute_midrank(pos_preds)
    ty = compute_midrank(neg_preds)
    tz = compute_midrank(predictions)
    
    auc = (np.sum(tz[labels==1]) - m*(m+1)/2) / (m*n)
    v01 = (tz[labels==1] - tx) / n
    v10 = 1 - (tz[labels==0] - ty) / m
    return auc, v01, v10

def delong_test(y, p1, p2):
    m = np.sum(y == 1)
    n = np.sum(y == 0)
    auc1, v01_1, v10_1 = delong_components(p1, y)
    auc2, v01_2, v10_2 = delong_components(p2, y)
    
    cov01 = np.cov(v01_1, v01_2, ddof=1)[0, 1]
    cov10 = np.cov(v10_1, v10_2, ddof=1)[0, 1]
    var1 = np.var(v01_1, ddof=1)/m + np.var(v10_1, ddof=1)/n
    var2 = np.var(v01_2, ddof=1)/m + np.var(v10_2, ddof=1)/n
    cov = cov01/m + cov10/n
    
    var_diff = max(var1 + var2 - 2 * cov, 1e-12)
    z = (auc1 - auc2) / np.sqrt(var_diff)
    p = 2 * stats.norm.sf(abs(z))
    return auc1, auc2, z, p

def load(path):
    df = pd.read_csv(path)
    return df.y_true.to_numpy().astype(int), df.y_prob.to_numpy().astype(float)

if __name__ == "__main__":
    # ---- HAM internal ----
    yD, pD = load(r".\results\runs\deep_baseline\ham_test_predictions_deep.csv")
    yH, pH = load(r".\results\runs\hybrid\ham_test_predictions_hybrid.csv")

    auc1, auc2, z, p = delong_test(yD, pD, pH)
    print("HAM internal:")
    print(f"Deep AUC={auc1:.4f}, Hybrid AUC={auc2:.4f}")
    print(f"z={z:.4f}, p-value={p:.6f}")
    print()

    # ---- ISIC external ----
    yD2, pD2 = load(r".\results\runs\deep_baseline\isic_task3_predictions_deep.csv")
    yH2, pH2 = load(r".\results\runs\hybrid\isic_task3_predictions_hybrid.csv")

    auc1e, auc2e, ze, pe = delong_test(yD2, pD2, pH2)
    print("ISIC external:")
    print(f"Deep AUC={auc1e:.4f}, Hybrid AUC={auc2e:.4f}")
    print(f"z={ze:.4f}, p-value={pe:.6f}")
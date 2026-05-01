import numpy as np
from src.evaluation.bootstrap import bootstrap_ci
from src.evaluation.metrics import compute_basic

def test_clustered_bootstrap():
    # Synthetic dataset: 100 lesions, each with 5 identical images.
    # Independent bootstrap should underestimate variance.
    # Clustered bootstrap should yield wider CIs.
    
    np.random.seed(42)
    n_lesions = 50
    images_per_lesion = 5
    
    # Generate lesion-level true labels and scores
    lesion_y = np.random.randint(0, 2, n_lesions)
    # score = true_label + noise
    lesion_p = lesion_y + np.random.normal(0, 0.5, n_lesions)
    lesion_p = 1 / (1 + np.exp(-lesion_p)) # to [0, 1]
    
    # Duplicate for clustered dataset
    y = np.repeat(lesion_y, images_per_lesion)
    p = np.repeat(lesion_p, images_per_lesion)
    groups = np.repeat(np.arange(n_lesions), images_per_lesion)
    
    # Calculate CIs
    n_resamples = 500
    lo_indep, hi_indep = bootstrap_ci(y, p, "AUC", n=n_resamples, seed=42, groups=None)
    lo_clust, hi_clust = bootstrap_ci(y, p, "AUC", n=n_resamples, seed=42, groups=groups)
    
    print(f"Independent Bootstrap CI (95%): [{lo_indep:.4f}, {hi_indep:.4f}], Width: {hi_indep - lo_indep:.4f}")
    print(f"Clustered Bootstrap CI (95%):   [{lo_clust:.4f}, {hi_clust:.4f}], Width: {hi_clust - lo_clust:.4f}")
    
    if (hi_clust - lo_clust) > (hi_indep - lo_indep) * 1.5:
        print("Success: Clustered bootstrap correctly identified much higher variance due to clustering.")
    else:
        print("Warning: Clustered bootstrap width not significantly larger than independent (check sample size/noise).")

if __name__ == "__main__":
    test_clustered_bootstrap()

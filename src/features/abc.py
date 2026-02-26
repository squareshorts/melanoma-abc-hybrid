import numpy as np
import cv2
from skimage.measure import perimeter
from skimage.feature import graycomatrix, graycoprops

def asymmetry(mask):
    m = (mask > 0).astype(np.uint8)
    if m.sum() < 10:
        return 1.0
    flip = np.fliplr(m)
    diff = np.abs(m.astype(np.int16) - flip.astype(np.int16))
    return float(diff.sum() / (m.sum() + 1e-8))

def border_irregularity(mask):
    m = (mask > 0).astype(np.uint8)
    a = float((m > 0).sum())
    if a < 10:
        return 0.0
    p = perimeter(m)
    return float(p / (2.0 * np.sqrt(np.pi * a) + 1e-8))

def color_stats(rgb, mask):
    m = mask > 0
    if m.sum() < 10:
        return {"color_mean_v": 0.0, "color_std_v": 0.0, "color_mean_s": 0.0, "color_std_s": 0.0}
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    v = hsv[..., 2][m].astype(np.float32)
    s = hsv[..., 1][m].astype(np.float32)
    return {
        "color_mean_v": float(v.mean()),
        "color_std_v": float(v.std()),
        "color_mean_s": float(s.mean()),
        "color_std_s": float(s.std()),
    }

def texture_glcm(rgb, mask):
    m = mask > 0
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    if m.sum() < 50:
        return {"glcm_contrast_d1": 0.0, "glcm_homogeneity_d1": 0.0, "glcm_contrast_d2": 0.0, "glcm_homogeneity_d2": 0.0}
    g = gray.copy()
    g[~m] = 0
    g = np.clip(g, 0, 255).astype(np.uint8)
    glcm = graycomatrix(g, distances=[1, 2], angles=[0], levels=256, symmetric=True, normed=True)
    return {
        "glcm_contrast_d1": float(graycoprops(glcm, "contrast")[0,0]),
        "glcm_homogeneity_d1": float(graycoprops(glcm, "homogeneity")[0,0]),
        "glcm_contrast_d2": float(graycoprops(glcm, "contrast")[1,0]),
        "glcm_homogeneity_d2": float(graycoprops(glcm, "homogeneity")[1,0]),
    }

def extract_abc(rgb, mask):
    feats = {
        "A_asymmetry": asymmetry(mask),
        "B_border_irreg": border_irregularity(mask),
    }
    feats.update({f"C_{k}": v for k, v in color_stats(rgb, mask).items()})
    feats.update({f"C_{k}": v for k, v in texture_glcm(rgb, mask).items()})
    return feats

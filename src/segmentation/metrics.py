import numpy as np

def dice(pred, gt):
    pred = (pred > 0).astype(np.uint8)
    gt = (gt > 0).astype(np.uint8)
    inter = (pred & gt).sum()
    return (2.0 * inter) / (pred.sum() + gt.sum() + 1e-8)

def iou(pred, gt):
    pred = (pred > 0).astype(np.uint8)
    gt = (gt > 0).astype(np.uint8)
    inter = (pred & gt).sum()
    union = (pred | gt).sum()
    return inter / (union + 1e-8)

def failure(mask, min_area_ratio=0.01):
    area = (mask > 0).sum()
    return area < (mask.size * float(min_area_ratio))

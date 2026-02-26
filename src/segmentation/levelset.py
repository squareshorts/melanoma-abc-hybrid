import numpy as np
import cv2
from skimage.segmentation import morphological_chan_vese

def segment_levelset(rgb, iterations=200, smoothing=1):
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
    gray = (gray - gray.min()) / (gray.max() - gray.min() + 1e-8)

    h, w = gray.shape
    init = np.zeros_like(gray, dtype=np.uint8)
    init[h//4:3*h//4, w//4:3*w//4] = 1

    mask = morphological_chan_vese(
        gray,
        num_iter=int(iterations),
        init_level_set=init,
        smoothing=int(smoothing),
    )
    return mask.astype(np.uint8)

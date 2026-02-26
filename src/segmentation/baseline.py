import numpy as np
import cv2

def segment_otsu(rgb):
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    blur = cv2.GaussianBlur(gray, (5,5), 0)
    _, th = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    m1 = (th == 0).astype(np.uint8)
    m2 = (th > 0).astype(np.uint8)
    return m1 if m1.sum() < m2.sum() else m2

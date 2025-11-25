# MINUTIAE: Thinning, crossing-number detection, grid encoding

import numpy as np
from skimage import morphology
import cv2
from .config import MINUTIAE_GRID

#  Thinning to skeleton

def thin_image(binary_img):
    skeleton = morphology.skeletonize(binary_img > 0)
    return skeleton.astype(np.uint8)

# Crossing-number minutiae detection

def detect_minutiae(skeleton, border_margin=5, min_dist=4):
    h, w = skeleton.shape
    padded = np.pad(skeleton, ((1, 1), (1, 1)), mode="constant", constant_values=0)
    raw_minutiae = []
    for y in range(1, h + 1):
        for x in range(1, w + 1):
            if padded[y, x] == 0:
                continue
            if y - 1 < border_margin or y - 1 >= h - border_margin:
                continue
            if x - 1 < border_margin or x - 1 >= w - border_margin:
                continue
            neighbors = np.array(
                [
                    padded[y - 1, x],
                    padded[y - 1, x + 1],
                    padded[y, x + 1],
                    padded[y + 1, x + 1],
                    padded[y + 1, x],
                    padded[y + 1, x - 1],
                    padded[y, x - 1],
                    padded[y - 1, x - 1],
                ],
                dtype=np.int32,
            )
            diffs = np.abs(neighbors - np.roll(neighbors, -1))
            cn = int(diffs.sum() // 2)
            m_type = None
            if cn == 1:
                m_type = "ending"
            elif cn == 3:
                m_type = "bifurcation"
            if m_type is not None:
                raw_minutiae.append({"y": y - 1, "x": x - 1, "type": m_type})
    filtered = []
    for m in raw_minutiae:
        keep = True
        for f in filtered:
            if m["type"] != f["type"]:
                continue
            dy = m["y"] - f["y"]
            dx = m["x"] - f["x"]
            if dx * dx + dy * dy < min_dist * min_dist:
                keep = False
                break
        if keep:
            filtered.append(m)
    return filtered

# Minutiae grid encoding (8×8×2)

def encode_minutiae_vector(minutiae_list, img_shape, grid=MINUTIAE_GRID):
    h, w = img_shape
    cell_h = h / grid
    cell_w = w / grid
    feat = np.zeros((grid, grid, 2), dtype=np.float32)
    for m in minutiae_list:
        y, x = m["y"], m["x"]
        r = int(y / cell_h)
        c = int(x / cell_w)
        if 0 <= r < grid and 0 <= c < grid:
            if m["type"] == "ending":
                feat[r, c, 0] += 1
            else:
                feat[r, c, 1] += 1
    vec = feat.flatten()
    if vec.max() > 0:
        vec = vec / vec.max()
    return vec


# Full minutiae feature extraction pipeline

def extract_minutiae_features(enhanced_img):
    smooth = cv2.GaussianBlur(enhanced_img, (3, 3), 0)
    _, binary = cv2.threshold(
        smooth, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )
    if np.mean(binary) > 127:
        binary = 255 - binary
    skeleton = thin_image(binary)
    minutiae_list = detect_minutiae(skeleton)
    feat_vec = encode_minutiae_vector(minutiae_list, skeleton.shape)
    return feat_vec.astype(np.float32)

# TEXTURE FEATURES: HOG + Wavelet feature extraction


import numpy as np
from skimage.feature import hog
import pywt

from config import HOG_PPC, HOG_CPB


# HOG + Wavelet features
def extract_hog_features(enhanced_img):
    hog_vec = hog(
        enhanced_img,
        orientations=9,
        pixels_per_cell=(HOG_PPC, HOG_PPC),
        cells_per_block=(HOG_CPB, HOG_CPB),
        block_norm="L2-Hys",
        visualize=False,
        feature_vector=True,
    )
    return hog_vec.astype(np.float32)


def extract_wavelet_features(enhanced_img, wavelet="db4", level=3):
    coeffs = pywt.wavedec2(enhanced_img, wavelet=wavelet, level=level)
    features = []
    for i, c in enumerate(coeffs):
        if i == 0:
            cA = c
            features.extend([np.mean(cA), np.std(cA), np.sum(cA ** 2)])
        else:
            cH, cV, cD = c
            for sb in (cH, cV, cD):
                features.extend([np.mean(sb), np.std(sb), np.sum(sb ** 2)])
    return np.array(features, dtype=np.float32)

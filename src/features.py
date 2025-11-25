# FEATURE FUSION: Minutiae + HOG + Wavelet

import numpy as np
from .preprocess import enhance_pipeline
from .minutiae import extract_minutiae_features
from .hog_wavelet import extract_hog_features, extract_wavelet_features

#  Extract fused features for a single image

def extract_features_for_image(path: str):
    enhanced = enhance_pipeline(path)
    minutiae_vec = extract_minutiae_features(enhanced)
    hog_vec = extract_hog_features(enhanced)
    wav_vec = extract_wavelet_features(enhanced)
    feat = np.concatenate([minutiae_vec, hog_vec, wav_vec], axis=0)
    return feat


# Build a feature matrix for a dataframe of images

def build_feature_matrix(df, label_col="hand", tag="hand_class"):
    X = []
    y = []
    n = len(df)
    for i, row in df.iterrows():
        path = row["path"]
        label = row[label_col]
        feat = extract_features_for_image(path)
        X.append(feat)
        y.append(label)
        if (i + 1) % 50 == 0 or (i + 1) == n:
            print(f"[{tag}] Processed {i + 1}/{n} images...")
    X = np.array(X, dtype=np.float32)
    y = np.array(y)
    return X, y

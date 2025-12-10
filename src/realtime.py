# REALTIME: Single-image handedness prediction using trained model

import os
import numpy as np
import matplotlib.pyplot as plt

from preprocess import enhance_pipeline
from minutiae import extract_minutiae_features
from hog_wavelet import extract_hog_features, extract_wavelet_features

plt.style.use("seaborn-v0_8-darkgrid")


# REAL-TIME HAND CLASSIFICATION (using fusion features)
def realtime_hand_predict_fusion(
    img_path: str,
    model_results,
    visualize: bool = False,
):
    if not os.path.exists(img_path):
        raise FileNotFoundError(f"Image not found: {img_path}")

    enhanced = enhance_pipeline(img_path)

    if visualize:
        plt.figure(figsize=(4, 4))
        plt.title("Enhanced Image (Realtime)")
        plt.imshow(enhanced, cmap="gray")
        plt.axis("off")
        plt.show()

    minutiae_vec = extract_minutiae_features(enhanced)
    hog_vec = extract_hog_features(enhanced)
    wav_vec = extract_wavelet_features(enhanced)

    feat = np.concatenate([minutiae_vec, hog_vec, wav_vec]).astype(np.float32)

    scaler = model_results["scaler"]
    pca = model_results["pca"]
    svm = model_results["svm"]

    feat_std = scaler.transform(feat.reshape(1, -1))
    feat_pca = pca.transform(feat_std)

    pred_label = svm.predict(feat_pca)[0]

    if hasattr(svm, "decision_function"):
        margin = svm.decision_function(feat_pca)
        margin_val = float(np.max(margin))
        confidence = 1.0 / (1.0 + np.exp(-abs(margin_val)))
    else:
        margin_val = 0.0
        confidence = 0.5

    print(
        f"[REALTIME FUSION] {os.path.basename(img_path)} → "
        f"{pred_label} (margin={margin_val:.3f}, conf≈{confidence:.3f})"
    )

    return {
        "predicted_label": pred_label,
        "margin": margin_val,
        "confidence": confidence,
        "image_path": img_path,
    }


def realtime_hand_demo_fusion(img_path, model_results, visualize=False):
    return realtime_hand_predict_fusion(
        img_path=img_path,
        model_results=model_results,
        visualize=visualize,
    )

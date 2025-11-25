# REALTIME: Single-image handedness prediction using trained model

import os
import numpy as np
from .features import extract_features_for_image

# 1) Core realtime prediction helper

def realtime_hand_predict(img_path, model_results):
    svm = model_results["svm"]
    scaler = model_results["scaler"]
    pca = model_results["pca"]
    feat = extract_features_for_image(img_path)
    feat = feat.reshape(1, -1)
    feat_std = scaler.transform(feat)
    feat_pca = pca.transform(feat_std)
    probs = svm.predict_proba(feat_pca)[0]
    pred_label = svm.classes_[np.argmax(probs)]
    margin = float(np.max(probs) - np.min(probs))
    confidence = float(np.max(probs))
    return {
        "predicted_label": pred_label,
        "margin": margin,
        "confidence": confidence,
        "image_path": img_path,
    }


# Console demo wrapper for realtime fusion

def realtime_fusion_demo(img_path, model_results):
    res = realtime_hand_predict(img_path, model_results)
    print(
        f"[REALTIME FUSION] {os.path.basename(img_path)} → {res['predicted_label']} "
        f"(margin={res['margin']:.3f}, conf≈{res['confidence']:.3f})"
    )
    return res

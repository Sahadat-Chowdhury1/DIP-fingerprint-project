# Classification on SOCOFing 
## Introduction

This repository provides the source code and experimental pipeline for a **traditional feature fusion–based fingerprint handedness classifier** (Left vs Right) using the **SOCOFing** dataset.

In many biometric systems, knowing whether a fingerprint belongs to the **left** or **right** hand can improve indexing, matching efficiency, and robustness. Instead of using deep learning, this project focuses on a **non–deep learning pipeline** that combines three well-known traditional fingerprint descriptors:

 * Minutiae features
 * HOG texture features 
 * Wavelet (DWT) features 

These are extracted from Gabor‑enhanced fingerprint images, fused into a single high-dimensional feature vector, reduced using **PCA**, and finally classified using an **RBF-kernel SVM**. The model is evaluated in a **subject-wise split** setting and supports **real-time prediction** on single images.

The overall workflow of this project is:

1. Dataset metadata construction (SOCOFing)
2. Subject-wise train/val/test split
3. Image preprocessing & enhancement (CLAHE + Gabor)
4. Minutiae, HOG, and Wavelet feature extraction
5. Feature fusion + standardization + PCA
6. SVM training, validation, and testing
7. Real-time Left/Right prediction on new fingerprint images




# Overview

This system extracts and fuses **three traditional fingerprint feature types**:

 # 1. Minutiae Features  
Ending & bifurcation counts encoded into an 8×8 grid feature vector.

# 2. HOG Texture Features  
Histogram of Oriented Gradients for ridge–flow texture description.

# 3. Wavelet Features  
Energy & statistics from multi‑level DWT (db4 wavelet).

All three feature sets are concatenated → standardized → reduced using PCA → 
classified using an **RBF‑kernel Support Vector Machine (SVM)**.

---



# Project Structure

```
# Classification on SOCOFing Using Traditional Feature Fusion

This project implements a non–deep-learning pipeline to classify a fingerprint
image as coming from the Left or Right hand, using the
SOCOFing fingerprint dataset.

The system combines three traditional feature families:

- Minutiae-based structural features
- Histogram of Oriented Gradients (HOG) texture features
- Wavelet energy features (DWT)

These are fused into a single feature vector, reduced with PCA, and finally
classified using an RBF-kernel SVM.  
On the SOCOFing Real subset with a subject-wise split, the model reaches
around 88–89% test accuracy and supports realtime prediction for a single
image.


1. Project Structure

This repo is organized as a small Python package under `src/`:

text
project_root/
│
├─ src/
│  ├─ config.py          # Paths and global constants
│  ├─ dataset.py         # Build metadata & subject-wise train/val/test split
│  ├─ preprocess.py      # CLAHE, normalization, resizing, orientation, Gabor
│  ├─ gabor.py           # Gabor kernel construction (if separated)
│  ├─ minutiae.py        # Thinning, minutiae detection, grid encoding
│  ├─ hog_wavelet.py     # HOG feature + wavelet (DWT) feature extraction
│  ├─ features.py        # Full feature fusion for one image / dataset
│  ├─ model.py           # Train SVM+PCA on fused features (main training script)
│  └─ realtime.py        # Load trained model objects and run realtime prediction
│
└─ README.md

```

#  How to Run

# 1. Install dependencies
```
pip install numpy pandas scikit-learn opencv-python scikit-image pywavelets matplotlib
```

# 2. Set your dataset path  
Edit `config.py`:

```
BASE_SOCOF_REAL = r"C:\path\to\SOCOFing\Real"
BASE_SOCOF_ALTERED_EASY = r"..."
```

# 3. Train the model  
Run:

```
python model.py
```

# 4. Realtime Prediction  
Use:


python realtime.py


# Output Example (Realtime)

[REALTIME FUSION] 100__M_Left_index_finger.BMP → Left
Confidence ≈ 0.87


# Credits
- Dataset: **SOCOFing (Sokoto Coventry Fingerprint Dataset)**  
- Author: Md Sahadatunnobi Chowdhury  



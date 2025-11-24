
import cv2
import numpy as np
from .config import (
    TARGET_SIZE,
    GABOR_KSIZE,
    GABOR_SIGMA,
    GABOR_LAMBDA,
    GABOR_GAMMA,
    GABOR_NUM_ORI,
)

def load_image_gray(path: str):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {path}")
    return img

def apply_clahe(img, clip_limit=2.0, tile_grid_size=(8, 8)):
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    return clahe.apply(img)

def normalize_image(img):
    img = img.astype(np.float32)
    img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)
    return img.astype(np.uint8)

def resize_image(img, target_size=TARGET_SIZE):
    if target_size is None:
        return img
    h, w = target_size
    return cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)

def preprocess_image(path: str):
    img = load_image_gray(path)
    img = apply_clahe(img)
    img = normalize_image(img)
    img = resize_image(img, TARGET_SIZE)
    return img

def estimate_orientation(img, block_size=16):
    img = img.astype(np.float32)
    gx = cv2.Sobel(img, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(img, cv2.CV_32F, 0, 1, ksize=3)
    h, w = img.shape
    orientation = np.zeros_like(img, dtype=np.float32)
    for i in range(0, h, block_size):
        for j in range(0, w, block_size):
            block_gx = gx[i : i + block_size, j : j + block_size].flatten()
            block_gy = gy[i : i + block_size, j : j + block_size].flatten()
            if block_gx.size == 0:
                continue
            vxx = np.sum(block_gx**2 - block_gy**2)
            vxy = 2 * np.sum(block_gx * block_gy)
            theta = 0.5 * np.arctan2(vxy, vxx + 1e-8)
            orientation[i : i + block_size, j : j + block_size] = theta
    return orientation

def build_gabor_kernels():
    kernels = []
    thetas = []
    for n in range(GABOR_NUM_ORI):
        theta = n * np.pi / GABOR_NUM_ORI
        kernel = cv2.getGaborKernel(
            (GABOR_KSIZE, GABOR_KSIZE),
            GABOR_SIGMA,
            theta,
            GABOR_LAMBDA,
            GABOR_GAMMA,
            0,
            ktype=cv2.CV_32F,
        )
        kernels.append(kernel)
        thetas.append(theta)
    return kernels, np.array(thetas, dtype=np.float32)

GABOR_KERNELS, GABOR_THETAS = build_gabor_kernels()

def gabor_enhance_oriented(img, orientation, block_size=16):
    img_f = img.astype(np.float32)
    h, w = img.shape
    enhanced = np.zeros_like(img_f)
    for i in range(0, h, block_size):
        for j in range(0, w, block_size):
            block = img_f[i : i + block_size, j : j + block_size]
            if block.size == 0:
                continue
            block_ori = orientation[i : i + block_size, j : j + block_size]
            avg_ori = np.mean(block_ori)
            idx = int(
                np.argmin(
                    np.abs(((GABOR_THETAS - avg_ori + np.pi) % (2 * np.pi)) - np.pi)
                )
            )
            kernel = GABOR_KERNELS[idx]
            filtered = cv2.filter2D(block, cv2.CV_32F, kernel)
            blended = 0.7 * block + 0.3 * filtered
            enhanced[i : i + block_size, j : j + block_size] = blended
    enhanced = cv2.normalize(enhanced, None, 0, 255, cv2.NORM_MINMAX)
    return enhanced.astype(np.uint8)

def enhance_pipeline(path: str):
    pre = preprocess_image(path)
    ori = estimate_orientation(pre)
    enhanced = gabor_enhance_oriented(pre, ori)
    return enhanced

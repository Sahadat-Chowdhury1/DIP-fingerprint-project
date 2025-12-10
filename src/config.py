
from pathlib import Path

# Global config and paths
RANDOM_SEED = 42

BASE_SOCOF_REAL = Path(r"C:\New folder\Digital image processing\SOCOFing\Real")
BASE_SOCOF_ALTERED_EASY = Path(r"C:\New folder\Digital image processing\SOCOFing\Altered-Easy")
BASE_SOCOF_ALTERED_MEDIUM = Path(r"C:\New folder\Digital image processing\SOCOFing\Altered-Medium")
BASE_SOCOF_ALTERED_HARD = Path(r"C:\New folder\Digital image processing\SOCOFing\Altered-Hard")

TARGET_SIZE = (256, 256)

# Gabor parameters
GABOR_KSIZE = 11
GABOR_SIGMA = 4.0
GABOR_LAMBDA = 8.0
GABOR_GAMMA = 0.5
GABOR_NUM_ORI = 8

# HOG parameters
HOG_PPC = 16
HOG_CPB = 2

# Minutiae encoding grid
MINUTIAE_GRID = 8

# PCA explained variance target
PCA_VARIANCE = 0.98

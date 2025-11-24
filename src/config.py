
from pathlib import Path
import random
import numpy as np

RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

BASE_SOCOF_REAL = Path(r"C:\New folder\Digital image processing\SOCOFing\Real")
BASE_SOCOF_ALTERED_EASY = Path(r"C:\New folder\Digital image processing\SOCOFing\Altered-Easy")
BASE_SOCOF_ALTERED_MEDIUM = Path(r"C:\New folder\Digital image processing\SOCOFing\Altered-Medium")
BASE_SOCOF_ALTERED_HARD = Path(r"C:\New folder\Digital image processing\SOCOFing\Altered-Hard")

TARGET_SIZE = (256, 256)

GABOR_KSIZE = 11
GABOR_SIGMA = 4.0
GABOR_LAMBDA = 8.0
GABOR_GAMMA = 0.5
GABOR_NUM_ORI = 8

HOG_PPC = 16
HOG_CPB = 2

MINUTIAE_GRID = 8

PCA_VARIANCE = 0.98

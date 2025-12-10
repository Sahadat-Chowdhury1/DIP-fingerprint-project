# End-to-end training + example realtime demo

import numpy as np
import pandas as pd

from config import RANDOM_SEED
from dataset import build_socof_metadata, subject_wise_split
from model import train_hand_classifier_svm_with_fusion
from realtime import realtime_hand_demo_fusion

import random

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)


# Main experiment entry point
def main():
    print("Building SOCOFing metadata...")
    df_meta = build_socof_metadata()
    print(df_meta.head())
    print(
        "Total images:",
        len(df_meta),
        "Unique subjects:",
        df_meta["subject_id"].nunique(),
    )

    df_real = df_meta[df_meta["alteration_level"] == "real"].reset_index(drop=True)
    print("Using only REAL images for hand classification.")
    df_train, df_val, df_test = subject_wise_split(df_real)

    results = train_hand_classifier_svm_with_fusion(df_train, df_val, df_test)

    print(
        "\n[FINAL FUSION] Hand classification test accuracy:",
        results["test_acc"],
    )
    print(
        "[FINAL FUSION] Best SVM params (C, gamma):",
        results["best_params"],
    )
    print(
        "[FINAL FUSION] Best validation accuracy:",
        results["best_val_acc"],
    )
    print("[FINAL FUSION] ROC AUC:", results["roc_auc"])

    example_path = df_test["path"].iloc[0]
    print("\nRunning a realtime demo on:", example_path)
    _ = realtime_hand_demo_fusion(example_path, model_results=results, visualize=False)


if __name__ == "__main__":
    main()


# End-to-end training + example realtime demo

import matplotlib.pyplot as plt
from src.dataset import build_socof_metadata, subject_wise_split
from src.model import train_hand_classifier_svm_with_fusion
from src.realtime import realtime_fusion_demo

# Main orchestration function

def main():
    print("Building SOCOFing metadata...")
    df_meta = build_socof_metadata()
    print(df_meta.head())
    print("Total images:", len(df_meta), "Unique subjects:", df_meta["subject_id"].nunique())

    df_real = df_meta[df_meta["alteration_level"] == "real"].reset_index(drop=True)
    print("Using only REAL images for hand classification.")
    df_train, df_val, df_test = subject_wise_split(df_real)

    results = train_hand_classifier_svm_with_fusion(df_train, df_val, df_test)

    print("\n[FINAL FUSION] Hand classification test accuracy:", results["test_acc"])
    print("[FINAL FUSION] Best SVM params (C, gamma):", results["best_params"])
    print("[FINAL FUSION] Best validation accuracy:", results["best_val_acc"])

    # Optional realtime demo on a sample image
    try:
        sample_path = df_real.iloc[0]["path"]
        print("\nRunning realtime demo on:", sample_path)
        realtime_fusion_demo(sample_path, results)
    except Exception as e:
        print("Realtime demo skipped:", e)

# Script entry guard

if __name__ == "__main__":
    main()

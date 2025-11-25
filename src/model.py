# MODEL: PCA + RBF-SVM training for handedness classification

import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import classification_report, accuracy_score
from sklearn.svm import SVC
from .features import build_feature_matrix
from .config import PCA_VARIANCE, RANDOM_SEED

# Train SVM with fused features and PCA

def train_hand_classifier_svm_with_fusion(df_train, df_val, df_test):
    # Extract fused features for train / val / test
    print("\n[Hand CLS SVM FUSION] Extracting TRAIN features.")
    X_train_raw, y_train = build_feature_matrix(df_train, label_col="hand", tag="hand_class")

    print("\n[Hand CLS SVM FUSION] Extracting VAL features.")
    X_val_raw, y_val = build_feature_matrix(df_val, label_col="hand", tag="hand_class")

    print("\n[Hand CLS SVM FUSION] Extracting TEST features.")
    X_test_raw, y_test = build_feature_matrix(df_test, label_col="hand", tag="hand_class")

    print("\nShapes (raw features):")
    print("  X_train_raw:", X_train_raw.shape)
    print("  X_val_raw  :", X_val_raw.shape)
    print("  X_test_raw :", X_test_raw.shape)

    # Standardize features and apply PCA on train/val/test
    scaler = StandardScaler()
    X_train_std = scaler.fit_transform(X_train_raw)
    X_val_std = scaler.transform(X_val_raw)
    X_test_std = scaler.transform(X_test_raw)

    pca = PCA(n_components=PCA_VARIANCE, svd_solver="full")
    X_train_pca = pca.fit_transform(X_train_std)
    X_val_pca = pca.transform(X_val_std)
    X_test_pca = pca.transform(X_test_std)

    print(f"\nHand CLS PCA output dimension: {X_train_pca.shape[1]}")

    #Hyperparameter tuning for SVM on validation set
    print("\n[Hand CLS SVM FUSION] Tuning SVM hyperparameters on validation set.")
    C_grid = [0.1, 1, 10]
    gamma_grid = ["scale", 0.001, 0.0001]

    best_val_acc = -1.0
    best_params = None

    for C in C_grid:
        for gamma in gamma_grid:
            print(f"  Trying SVM(C={C}, gamma={gamma}) .")
            svm = SVC(
                C=C,
                kernel="rbf",
                gamma=gamma,
                class_weight="balanced",
                probability=False,
                random_state=RANDOM_SEED,
            )
            svm.fit(X_train_pca, y_train)
            val_pred = svm.predict(X_val_pca)
            val_acc = accuracy_score(y_val, val_pred)
            print(f"    → val_acc = {val_acc:.4f}")
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_params = (C, gamma)

    print(
        f"\n[Hand CLS SVM FUSION] Best params from validation: "
        f"C={best_params[0]}, gamma={best_params[1]} (val_acc={best_val_acc:.4f})"
    )

    # Retrain SVM on train+val using best hyperparameters and evaluate on test
    X_full_raw = np.vstack([X_train_raw, X_val_raw])
    y_full = np.concatenate([y_train, y_val])

    scaler_full = StandardScaler()
    X_full_std = scaler_full.fit_transform(X_full_raw)
    X_test_std_final = scaler_full.transform(X_test_raw)

    pca_full = PCA(n_components=PCA_VARIANCE, svd_solver="full")
    X_full_pca = pca_full.fit_transform(X_full_std)
    X_test_pca_final = pca_full.transform(X_test_std_final)

    svm_final = SVC(
        C=best_params[0],
        kernel="rbf",
        gamma=best_params[1],
        class_weight="balanced",
        probability=True,
        random_state=RANDOM_SEED,
    )
    svm_final.fit(X_full_pca, y_full)

    test_pred = svm_final.predict(X_test_pca_final)
    test_acc = accuracy_score(y_test, test_pred)

    print(f"\n[Hand CLS SVM FUSION] Train(full) size: {len(y_full)}")
    print(f"[Hand CLS SVM FUSION] Test accuracy:        {test_acc:.4f}\n")
    print("[Hand CLS SVM FUSION] Classification report (test):")
    print(classification_report(y_test, test_pred))

    return {
        "scaler": scaler_full,
        "pca": pca_full,
        "svm": svm_final,
        "X_test_pca": X_test_pca_final,
        "y_test": y_test,
        "y_pred": test_pred,
        "test_acc": test_acc,
        "best_params": best_params,
        "best_val_acc": best_val_acc,
    }

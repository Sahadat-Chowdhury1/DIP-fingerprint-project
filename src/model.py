# MODEL: PCA + RBF-SVM training for handedness classification

import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import (
    classification_report,
    accuracy_score,
    confusion_matrix,
    roc_curve,
    auc,
)
from sklearn.svm import SVC

from config import PCA_VARIANCE, RANDOM_SEED
from features import build_feature_matrix

plt.style.use("seaborn-v0_8-darkgrid")


# Train SVM (RBF) + PCA + Confusion Matrix + ROC
def train_hand_classifier_svm_with_fusion(df_train, df_val, df_test):
    print("\n[Hand CLS SVM FUSION] Extracting TRAIN features...")
    X_train_raw, y_train = build_feature_matrix(
        df_train, label_col="hand", tag="hand_class"
    )

    print("\n[Hand CLS SVM FUSION] Extracting VAL features...")
    X_val_raw, y_val = build_feature_matrix(
        df_val, label_col="hand", tag="hand_class"
    )

    print("\n[Hand CLS SVM FUSION] Extracting TEST features...")
    X_test_raw, y_test = build_feature_matrix(
        df_test, label_col="hand", tag="hand_class"
    )

    print("\nShapes (raw features):")
    print("  X_train_raw:", X_train_raw.shape)
    print("  X_val_raw  :", X_val_raw.shape)
    print("  X_test_raw :", X_test_raw.shape)

    scaler = StandardScaler()
    X_train_std = scaler.fit_transform(X_train_raw)
    X_val_std = scaler.transform(X_val_raw)
    X_test_std = scaler.transform(X_test_raw)

    pca = PCA(n_components=PCA_VARIANCE, svd_solver="full")
    X_train_pca = pca.fit_transform(X_train_std)
    X_val_pca = pca.transform(X_val_std)
    X_test_pca = pca.transform(X_test_std)

    print(f"\nHand CLS PCA output dimension: {X_train_pca.shape[1]}")

    print("\n[Hand CLS SVM FUSION] Tuning SVM hyperparameters on validation set...")
    C_grid = [0.1, 1, 10]
    gamma_grid = ["scale", 0.001, 0.0001]

    best_val_acc = -1.0
    best_params = None

    for C in C_grid:
        for gamma in gamma_grid:
            print(f"  Trying SVM(C={C}, gamma={gamma}) ...")
            svm = SVC(
                C=C,
                kernel="rbf",
                gamma=gamma,
                class_weight="balanced",
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

    X_full_raw = np.vstack([X_train_raw, X_val_raw])
    y_full = np.concatenate([y_train, y_val])

    X_full_std = scaler.fit_transform(X_full_raw)
    X_test_std_final = scaler.transform(X_test_raw)

    pca_full = PCA(n_components=PCA_VARIANCE, svd_solver="full")
    X_full_pca = pca_full.fit_transform(X_full_std)
    X_test_pca_final = pca_full.transform(X_test_std_final)

    svm_final = SVC(
        C=best_params[0],
        kernel="rbf",
        gamma=best_params[1],
        class_weight="balanced",
        random_state=RANDOM_SEED,
    )
    svm_final.fit(X_full_pca, y_full)

    test_pred = svm_final.predict(X_test_pca_final)
    test_acc = accuracy_score(y_test, test_pred)
    print(f"\n[Hand CLS SVM FUSION] Train(full) size: {len(y_full)}")
    print(f"[Hand CLS SVM FUSION] Test accuracy:        {test_acc:.4f}\n")

    print("[Hand CLS SVM FUSION] Classification report (test):")
    print(classification_report(y_test, test_pred))

    cm = confusion_matrix(y_test, test_pred)
    class_names = np.unique(y_test)

    print("\n[Hand CLS SVM FUSION] Confusion matrix (counts):")
    print(cm)

    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    ax.figure.colorbar(im, ax=ax)

    ax.set(
        xticks=np.arange(cm.shape[1]),
        yticks=np.arange(cm.shape[0]),
        xticklabels=class_names,
        yticklabels=class_names,
        ylabel="True label",
        xlabel="Predicted label",
        title="Confusion Matrix – Handedness (Counts)",
    )

    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j,
                i,
                format(cm[i, j], "d"),
                ha="center",
                va="center",
                color="white" if cm[i, j] > thresh else "black",
            )

    fig.tight_layout()
    plt.show()

    cm_norm = cm.astype("float") / cm.sum(axis=1, keepdims=True)

    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm_norm, interpolation="nearest", cmap="Blues")
    ax.figure.colorbar(im, ax=ax)

    ax.set(
        xticks=np.arange(cm_norm.shape[1]),
        yticks=np.arange(cm_norm.shape[0]),
        xticklabels=class_names,
        yticklabels=class_names,
        ylabel="True label",
        xlabel="Predicted label",
        title="Normalized Confusion Matrix – Handedness",
    )

    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    for i in range(cm_norm.shape[0]):
        for j in range(cm_norm.shape[1]):
            ax.text(
                j,
                i,
                f"{cm_norm[i, j]:.2f}",
                ha="center",
                va="center",
                color="white" if cm_norm[i, j] > 0.5 else "black",
            )

    fig.tight_layout()
    plt.show()

    scores = svm_final.decision_function(X_test_pca_final)
    pos_label = class_names[1]

    fpr, tpr, _ = roc_curve(y_test, scores, pos_label=pos_label)
    roc_auc = auc(fpr, tpr)

    print(
        f"\n[Hand CLS SVM FUSION] ROC AUC (positive class = {pos_label}): {roc_auc:.4f}"
    )

    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(fpr, tpr, label=f"ROC curve (AUC = {roc_auc:.3f})")
    ax.plot([0, 1], [0, 1], linestyle="--", label="Random")
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve – Handedness Classification")
    ax.legend(loc="lower right")
    ax.grid(True)
    plt.show()

    return {
        "scaler": scaler,
        "pca": pca_full,
        "svm": svm_final,
        "X_test_pca": X_test_pca_final,
        "y_test": y_test,
        "y_pred": test_pred,
        "test_acc": test_acc,
        "best_params": best_params,
        "best_val_acc": best_val_acc,
        "confusion_matrix": cm,
        "confusion_matrix_normalized": cm_norm,
        "fpr": fpr,
        "tpr": tpr,
        "roc_auc": roc_auc,
    }

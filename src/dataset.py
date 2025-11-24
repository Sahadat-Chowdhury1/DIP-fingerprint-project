
import os
from pathlib import Path
import random
import pandas as pd
from .config import (
    BASE_SOCOF_REAL,
    BASE_SOCOF_ALTERED_EASY,
    BASE_SOCOF_ALTERED_MEDIUM,
    BASE_SOCOF_ALTERED_HARD,
)

def parse_socof_filename(filename: str):
    name = os.path.splitext(os.path.basename(filename))[0]
    parts = name.split("__")
    if len(parts) != 2:
        return None, None, None, None, None
    subj_str = parts[0]
    rest = parts[1]
    tokens = rest.split("_")
    if len(tokens) < 4:
        return None, None, None, None, None
    gender = tokens[0]
    hand = tokens[1]
    finger_name = "_".join(tokens[2:4])
    alteration_type = None
    if len(tokens) > 4:
        alteration_type = tokens[4]
    try:
        subject_id = int(subj_str)
    except ValueError:
        subject_id = None
    return subject_id, gender, hand, finger_name, alteration_type

def build_socof_metadata():
    records = []

    def scan_folder(base_path: Path, alteration_level: str):
        if not base_path.exists():
            print(f"[WARN] Missing folder: {base_path}, skipping...")
            return
        for root, dirs, files in os.walk(base_path):
            for f in files:
                if not f.lower().endswith((".bmp", ".png", ".jpg", ".jpeg", ".tif", ".tiff")):
                    continue
                full_path = Path(root) / f
                sid, gender, hand, finger_name, alt_type = parse_socof_filename(f)
                records.append(
                    {
                        "path": str(full_path),
                        "subject_id": sid,
                        "gender": gender,
                        "hand": hand,
                        "finger_name": finger_name,
                        "alteration_type": alt_type,
                        "alteration_level": alteration_level,
                    }
                )

    scan_folder(BASE_SOCOF_REAL, "real")
    scan_folder(BASE_SOCOF_ALTERED_EASY, "easy")
    scan_folder(BASE_SOCOF_ALTERED_MEDIUM, "medium")
    scan_folder(BASE_SOCOF_ALTERED_HARD, "hard")

    df = pd.DataFrame(records)
    return df

def subject_wise_split(df, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15):
    subjects = sorted(df["subject_id"].dropna().unique())
    random.shuffle(subjects)
    n = len(subjects)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    train_subj = subjects[:n_train]
    val_subj = subjects[n_train : n_train + n_val]
    test_subj = subjects[n_train + n_val :]
    df_train = df[df["subject_id"].isin(train_subj)].reset_index(drop=True)
    df_val = df[df["subject_id"].isin(val_subj)].reset_index(drop=True)
    df_test = df[df["subject_id"].isin(test_subj)].reset_index(drop=True)
    print(f"Train subjects: {len(train_subj)} Images: {len(df_train)}")
    print(f"Val subjects  : {len(val_subj)} Images: {len(df_val)}")
    print(f"Test subjects : {len(test_subj)} Images: {len(df_test)}")
    return df_train, df_val, df_test

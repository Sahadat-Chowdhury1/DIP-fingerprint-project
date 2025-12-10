# DATASET: SOCOFing metadata construction and subject-wise split

import os
import random
import pandas as pd
from pathlib import Path
from typing import Tuple

from config import (
    BASE_SOCOF_REAL,
    BASE_SOCOF_ALTERED_EASY,
    BASE_SOCOF_ALTERED_MEDIUM,
    BASE_SOCOF_ALTERED_HARD,
    RANDOM_SEED,
)

random.seed(RANDOM_SEED)


# SOCOFing filename parsing + metadata
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


def _scan_folder(base_path: Path, alteration_level: str, records: list):
    if not base_path.exists():
        print(f"[WARN] Missing folder: {base_path}, skipping...")
        return
    for root, _, files in os.walk(base_path):
        for f in files:
            if not f.lower().endswith(
                (".bmp", ".png", ".jpg", ".jpeg", ".tif", ".tiff")
            ):
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


def build_socof_metadata() -> pd.DataFrame:
    records = []

    _scan_folder(BASE_SOCOF_REAL, "real", records)
    _scan_folder(BASE_SOCOF_ALTERED_EASY, "easy", records)
    _scan_folder(BASE_SOCOF_ALTERED_MEDIUM, "medium", records)
    _scan_folder(BASE_SOCOF_ALTERED_HARD, "hard", records)

    df = pd.DataFrame(records)
    return df


# Subject-wise split (train / val / test)
def subject_wise_split(
    df: pd.DataFrame,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
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

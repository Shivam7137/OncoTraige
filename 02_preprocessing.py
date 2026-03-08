"""
p2_01_split.py — New Preprocessing (Rebuild Merge)
====================================================
Replaces old VIF → scale → winsorize pipeline.
New pipeline: stratified split only. All 30 features, raw.
"""

import numpy as np
import pandas as pd
import json
from sklearn.model_selection import train_test_split
from p1_00_config import (
    DATA_PATH, DROP_COLS, TARGET_COL,
    TEST_SIZE, RANDOM_STATE, CLASS_WEIGHT, THRESHOLD,
    ARTIFACT_DIR, INTER_DIR
)


def load_and_prepare():
    df = pd.read_csv(DATA_PATH)
    df = df.drop(columns=DROP_COLS, errors='ignore')
    df[TARGET_COL] = (df[TARGET_COL] == 'M').astype(int)
    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL]
    return X, y


if __name__ == '__main__':
    print('=' * 50)
    print('  p2 — PREPROCESSING (SPLIT ONLY)')
    print('=' * 50)

    X, y = load_and_prepare()

    # Save full feature list — all 30 features
    feature_names = list(X.columns)
    with open(f'{ARTIFACT_DIR}/all_features.json', 'w') as f:
        json.dump(feature_names, f, indent=2)

    # Stratified split — same random state as rebuild
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        stratify=y,
        random_state=RANDOM_STATE
    )

    # Save as numpy arrays
    np.save(f'{INTER_DIR}/X_train_final.npy', X_train.values)
    np.save(f'{INTER_DIR}/X_test_final.npy', X_test.values)
    np.save(f'{INTER_DIR}/y_train.npy', y_train.values)
    np.save(f'{INTER_DIR}/y_test.npy', y_test.values)

    # Save config artifacts
    with open(f'{ARTIFACT_DIR}/class_weights.json', 'w') as f:
        json.dump(CLASS_WEIGHT, f)
    with open(f'{ARTIFACT_DIR}/threshold.json', 'w') as f:
        json.dump({'threshold': THRESHOLD}, f)

    print(f'  Train: {X_train.shape}, Test: {X_test.shape}')
    print(f'  Train malignant %: {y_train.mean():.3f}')
    print(f'  Test  malignant %: {y_test.mean():.3f}')
    print(f'  Features saved: {len(feature_names)}')
    print(f'\n  Artifacts: all_features.json, class_weights.json, threshold.json')
    print(f'  Intermediate: X_train_final.npy, X_test_final.npy, y_train.npy, y_test.npy')

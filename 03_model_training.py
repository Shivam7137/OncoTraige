"""
p3_01_rf_final.py — Final RF Model (Rebuild Merge)
====================================================
Trains RF with class_weight={0:1,1:4}, threshold=0.20,
runs 5-fold CV, applies Platt scaling, saves all artifacts.
"""

import numpy as np
import json
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    roc_auc_score, brier_score_loss,
    recall_score, confusion_matrix
)
from sklearn.model_selection import StratifiedKFold, cross_validate
from p1_00_config import (
    N_ESTIMATORS, MAX_FEATURES, CLASS_WEIGHT, RANDOM_STATE,
    THRESHOLD, INTER_DIR, ARTIFACT_DIR
)

if __name__ == '__main__':
    print('=' * 50)
    print('  p3 — FINAL RF MODEL')
    print('=' * 50)

    # Load data
    X_train = np.load(f'{INTER_DIR}/X_train_final.npy')
    X_test  = np.load(f'{INTER_DIR}/X_test_final.npy')
    y_train = np.load(f'{INTER_DIR}/y_train.npy')
    y_test  = np.load(f'{INTER_DIR}/y_test.npy')

    print(f'  Train: {X_train.shape}, Test: {X_test.shape}')

    # ── TRAIN FINAL MODEL ──
    model = RandomForestClassifier(
        n_estimators=N_ESTIMATORS,
        max_features=MAX_FEATURES,
        class_weight=CLASS_WEIGHT,
        random_state=RANDOM_STATE,
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    print('  Model trained.')

    # ── CROSS-VALIDATION ──
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    cv_results = cross_validate(
        model, X_train, y_train,
        cv=cv,
        scoring={
            'auc': 'roc_auc',
            'sensitivity': 'recall',
            'brier': 'neg_brier_score'
        },
        return_train_score=False
    )

    print('\n  5-Fold CV Results:')
    print(f'    AUC         : {cv_results["test_auc"].mean():.4f} +/- {cv_results["test_auc"].std():.4f}')
    print(f'    Sensitivity : {cv_results["test_sensitivity"].mean():.4f} +/- {cv_results["test_sensitivity"].std():.4f}')
    print(f'    Brier       : {-cv_results["test_brier"].mean():.4f} +/- {cv_results["test_brier"].std():.4f}')

    # ── PLATT SCALING ──
    calibrated = CalibratedClassifierCV(model, method='sigmoid', cv=5)
    calibrated.fit(X_train, y_train)
    print('  Platt scaling complete.')

    # ── EVALUATE WITH THRESHOLD ──
    prob_test = calibrated.predict_proba(X_test)[:, 1]
    pred_test = (prob_test >= THRESHOLD).astype(int)

    tn, fp, fn, tp = confusion_matrix(y_test, pred_test).ravel()

    metrics = {
        'auc':         round(roc_auc_score(y_test, prob_test), 4),
        'brier':       round(brier_score_loss(y_test, prob_test), 4),
        'sensitivity': round(recall_score(y_test, pred_test), 4),
        'specificity': round(tn / (tn + fp), 4),
        'threshold':   THRESHOLD,
        'tp': int(tp), 'tn': int(tn), 'fp': int(fp), 'fn': int(fn),
        'cv_auc_mean':   round(cv_results['test_auc'].mean(), 4),
        'cv_sens_mean':  round(cv_results['test_sensitivity'].mean(), 4),
        'cv_brier_mean': round(-cv_results['test_brier'].mean(), 4),
    }

    print(f'\n  Test Results (threshold={THRESHOLD}):')
    for k, v in metrics.items():
        print(f'    {k:<20}: {v}')

    # ── SAVE ──
    joblib.dump(model,      f'{ARTIFACT_DIR}/final_model.joblib')
    joblib.dump(calibrated, f'{ARTIFACT_DIR}/final_model_calibrated.joblib')
    np.save(f'{INTER_DIR}/rf_y_prob_test.npy', prob_test)
    with open(f'{ARTIFACT_DIR}/rf_metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)

    print('\n  Artifacts saved:')
    print('    final_model.joblib, final_model_calibrated.joblib')
    print('    rf_y_prob_test.npy, rf_metrics.json')

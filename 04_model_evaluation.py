"""
p4_01_shap_reliability.py — SHAP + Reliability Score (Rebuild Merge)
=====================================================================
New formula: 0.50 × probability_margin + 0.30 × shap_spread + 0.20 × leaf_consensus
No external KNN model — uses RF's own internal tree structure.
"""

import numpy as np
import pandas as pd
import json
import joblib
import warnings
warnings.filterwarnings('ignore')

import shap
from scipy.stats import entropy as scipy_entropy
from p1_00_config import (
    THRESHOLD, INTER_DIR, ARTIFACT_DIR,
    REL_WEIGHT_MARGIN, REL_WEIGHT_SHAP, REL_WEIGHT_LEAF,
    REL_HIGH_THRESHOLD, REL_LOW_THRESHOLD
)


# ── Reliability Components ───────────────────────────────────────────

def probability_margin(prob):
    """Distance from 0.5, scaled to [0,1]."""
    return float(abs(prob - 0.5) / 0.5)


def shap_spread(patient_shap):
    """
    Normalized Shannon entropy of absolute SHAP values.
    High = prediction driven by many features (robust).
    Low  = one feature dominates (fragile).
    """
    abs_shap = np.abs(patient_shap)
    total = abs_shap.sum()
    if total == 0:
        return 1.0
    proportions = abs_shap / total
    return float(scipy_entropy(proportions + 1e-10) / np.log(len(patient_shap)))


def leaf_consensus(model, patient_array):
    """
    Fraction of trees that agree with the majority vote.
    Uses RF's own internal structure — no external model.
    """
    votes = np.array([t.predict(patient_array)[0] for t in model.estimators_])
    majority = votes.mean().round()
    return float((votes == majority).mean())


def compute_reliability(prob, patient_shap, model, patient_array):
    margin = probability_margin(prob)
    spread = shap_spread(patient_shap)
    leaf = leaf_consensus(model, patient_array)
    score = (REL_WEIGHT_MARGIN * margin +
             REL_WEIGHT_SHAP * spread +
             REL_WEIGHT_LEAF * leaf)
    return round(score, 4), round(margin, 4), round(spread, 4), round(leaf, 4)


def get_tier(score):
    if score >= REL_HIGH_THRESHOLD:
        return 'High'
    elif score >= REL_LOW_THRESHOLD:
        return 'Moderate'
    else:
        return 'Low'


def get_recommendation(score, prob, top_drivers):
    if abs(prob - 0.5) < 0.15:
        return 'Prediction near decision boundary. Escalate to Core Needle Biopsy (CNB).'
    if score >= REL_HIGH_THRESHOLD:
        return 'High reliability. Proceed with standard care plan.'
    elif score >= REL_LOW_THRESHOLD:
        drivers = ', '.join([d['feature'] for d in top_drivers])
        return f'Moderate reliability. Review: {drivers}. Consider re-measurement.'
    else:
        return 'Low reliability — conflicting signals. Escalate to Core Needle Biopsy (CNB).'


def get_top_drivers(patient_shap, features, n=3):
    df = pd.DataFrame({
        'feature': features,
        'shap_value': patient_shap,
        'abs_impact': np.abs(patient_shap)
    }).sort_values('abs_impact', ascending=False)
    return [
        {'feature': row['feature'],
         'impact': round(row['shap_value'], 4),
         'direction': 'toward malignant' if row['shap_value'] > 0 else 'toward benign'}
        for _, row in df.head(n).iterrows()
    ]


# ── Master triage function (for Streamlit) ───────────────────────────

def generate_triage_output(raw_feature_array, calibrated, model, explainer, features):
    """
    Input : numpy array of shape (30,) — raw unscaled feature values
    Output: dict with all triage information for Streamlit rendering
    """
    arr = np.array(raw_feature_array).reshape(1, -1)
    prob = float(calibrated.predict_proba(arr)[0][1])
    pred = 'Malignant' if prob >= THRESHOLD else 'Benign'

    shap_vals = explainer.shap_values(arr)
    if isinstance(shap_vals, list):
        shap_m = shap_vals[1][0]
    elif shap_vals.ndim == 3:
        shap_m = shap_vals[0, :, 1]
    else:
        shap_m = shap_vals[0]

    top_drivers = get_top_drivers(shap_m, features)
    score, margin, spread, leaf = compute_reliability(prob, shap_m, model, arr)
    tier = get_tier(score)
    rec = get_recommendation(score, prob, top_drivers)

    return {
        'prediction': pred,
        'probability': round(prob, 4),
        'threshold': THRESHOLD,
        'reliability': score,
        'tier': tier,
        'recommendation': rec,
        'top_drivers': top_drivers,
        'components': {
            'probability_margin': margin,
            'shap_spread': spread,
            'leaf_consensus': leaf
        },
        'shap_values': shap_m.tolist()
    }


# ── Main: compute for all test patients ──────────────────────────────

if __name__ == '__main__':
    print('=' * 50)
    print('  p4 — SHAP + RELIABILITY SCORE')
    print('=' * 50)

    # Load
    X_train   = np.load(f'{INTER_DIR}/X_train_final.npy')
    X_test    = np.load(f'{INTER_DIR}/X_test_final.npy')
    y_train   = np.load(f'{INTER_DIR}/y_train.npy')
    y_test    = np.load(f'{INTER_DIR}/y_test.npy')
    prob_test = np.load(f'{INTER_DIR}/rf_y_prob_test.npy')
    model     = joblib.load(f'{ARTIFACT_DIR}/final_model.joblib')
    calibrated = joblib.load(f'{ARTIFACT_DIR}/final_model_calibrated.joblib')
    features  = json.load(open(f'{ARTIFACT_DIR}/all_features.json'))

    # SHAP
    print('\n  Computing SHAP values (TreeExplainer)...')
    explainer = shap.TreeExplainer(model)
    shap_explanation = explainer(X_test)

    # Handle both old-style (list) and new-style (Explanation object)
    if hasattr(shap_explanation, 'values'):
        raw_vals = shap_explanation.values
        base_val = shap_explanation.base_values
    else:
        raw_vals = explainer.shap_values(X_test)
        base_val = explainer.expected_value

    # For binary classification RF, SHAP may return 3D
    if isinstance(raw_vals, list):
        shap_malignant = raw_vals[1]
    elif raw_vals.ndim == 3:
        shap_malignant = raw_vals[:, :, 1]
    else:
        shap_malignant = raw_vals

    # Handle base values
    if isinstance(base_val, np.ndarray) and base_val.ndim >= 1:
        if base_val.ndim == 2:
            base_val_save = base_val[0]
        elif base_val.ndim == 1 and len(base_val) == 2:
            base_val_save = base_val[1]
        else:
            base_val_save = base_val
    else:
        base_val_save = np.array([base_val])

    np.save(f'{ARTIFACT_DIR}/shap_values_test.npy', shap_malignant)
    np.save(f'{ARTIFACT_DIR}/shap_base_value.npy', base_val_save)
    print(f'  SHAP values shape: {shap_malignant.shape}')

    # Compute reliability for all test patients
    print('\n  Computing reliability scores...')
    records = []
    for i in range(len(X_test)):
        prob = prob_test[i]
        patient_arr = X_test[i].reshape(1, -1)
        score, margin, spread, leaf = compute_reliability(
            prob, shap_malignant[i], model, patient_arr
        )
        top_drivers = get_top_drivers(shap_malignant[i], features)

        records.append({
            'patient_idx': i,
            'probability': round(float(prob), 4),
            'reliability': score,
            'tier': get_tier(score),
            'margin': margin,
            'spread': spread,
            'leaf': leaf,
            'y_true': int(y_test[i]),
            'y_pred': int((prob >= THRESHOLD)),
            'recommendation': get_recommendation(score, prob, top_drivers)
        })

    rel_df = pd.DataFrame(records)
    rel_df.to_csv(f'{ARTIFACT_DIR}/reliability_scores_test.csv', index=False)

    # Summary
    print('\n  Reliability Score Summary:')
    for tier in ['High', 'Moderate', 'Low']:
        subset = rel_df[rel_df['tier'] == tier]
        if len(subset) > 0:
            acc = (subset['y_true'] == subset['y_pred']).mean()
            print(f'    {tier:<10}: {len(subset):>3} patients — accuracy {acc:.1%}')
        else:
            print(f'    {tier:<10}:   0 patients')

    print(f'\n  Artifacts saved:')
    print(f'    shap_values_test.npy, shap_base_value.npy')
    print(f'    reliability_scores_test.csv')
    print(f'\n  generate_triage_output() ready for Streamlit.')

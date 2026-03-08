# OncoTriage AI — Phase 3: Random Forest Plan

---

## Why Random Forest First

| Reason | What It Means for OncoTriage AI |
|---|---|
| Natively handles class imbalance | `class_weight='balanced'` is a first-class parameter |
| Tree-based = scale invariant | Works well with winsorized/scaled features |
| Best SHAP compatibility | TreeExplainer is the fastest, most accurate SHAP explainer |
| Robust to remaining correlations | Random feature subsampling handles moderate correlations gracefully |
| Low overfitting risk | Bagging self-regularizes — important on a 455-sample training set |

RF also serves as the baseline against which XGBoost is judged in Phase 3b.

---

## Step 0 — Load Phase 2 Artifacts

```python
import numpy as np
import json, joblib

X_train = np.load('outputs/intermediate/X_train_final.npy')
X_test  = np.load('outputs/intermediate/X_test_final.npy')
y_train = np.load('outputs/intermediate/y_train.npy')
y_test  = np.load('outputs/intermediate/y_test.npy')

scaler           = joblib.load('outputs/artifacts/scaler.joblib')
winsorize_bounds = json.load(open('outputs/artifacts/winsorize_bounds.json'))
class_weights    = json.load(open('outputs/artifacts/class_weights.json'))
features         = json.load(open('outputs/surviving_features.json'))

# Sanity check
assert X_train.shape == (455, 17), 'Shape mismatch'
assert X_test.shape  == (114, 17), 'Shape mismatch'
print('All Phase 2 artifacts loaded. Ready for modeling.')
```

> **If either assert fails — stop and debug Phase 2 before proceeding.**

---

## Step 1 — Baseline Model (No Tuning)

Establish a reference point before any hyperparameter tuning.

```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_validate

# JSON stores keys as strings — convert back to int
cw = {int(k): v for k, v in class_weights.items()}

rf_baseline = RandomForestClassifier(
    n_estimators = 100,
    class_weight = cw,        # {0: 0.796, 1: 1.346}
    random_state = 42,
    n_jobs       = -1
)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

scoring = {
    'sensitivity': 'recall',
    'roc_auc'    : 'roc_auc',
    'f1'         : 'f1',
    'brier'      : 'neg_brier_score'
}

baseline_results = cross_validate(
    rf_baseline, X_train, y_train,
    cv=cv, scoring=scoring, return_train_score=True
)

for metric, scores in baseline_results.items():
    if metric.startswith('test_'):
        print(f'{metric:30s}: {np.mean(scores):.4f} +/- {np.std(scores):.4f}')
```

### Target Thresholds

| Metric | Target | Clinical Meaning |
|---|---|---|
| Sensitivity (Recall — Malignant) | ≥ 0.93 | Catching 93%+ of real cancers |
| Specificity | ≥ 0.88 | Clearing 88%+ of benign cases |
| ROC-AUC | ≥ 0.97 | Strong discrimination |
| F1 Score | ≥ 0.93 | Precision/recall balance |
| Brier Score | ≤ 0.07 | Probability calibration quality |

> If baseline misses these targets, revisit the pipeline before tuning.

---

## Step 2 — Hyperparameter Tuning

### What to Tune

| Parameter | Search Range | Why |
|---|---|---|
| `n_estimators` | [100, 200, 300, 500] | More trees = more stable probability estimates |
| `max_depth` | [None, 5, 10, 15, 20] | Controls overfitting on 455 samples |
| `min_samples_leaf` | [1, 2, 4, 8] | Higher = better calibration |
| `max_features` | ['sqrt', 'log2', 0.5] | Controls tree diversity |

### RandomizedSearchCV (30 iterations)

```python
from sklearn.model_selection import RandomizedSearchCV

param_grid = {
    'n_estimators'    : [100, 200, 300, 500],
    'max_depth'       : [None, 5, 10, 15, 20],
    'min_samples_leaf': [1, 2, 4, 8],
    'max_features'    : ['sqrt', 'log2', 0.5]
}

rf_search = RandomizedSearchCV(
    estimator           = RandomForestClassifier(class_weight=cw, random_state=42, n_jobs=-1),
    param_distributions = param_grid,
    n_iter              = 30,
    scoring             = 'roc_auc',
    cv                  = cv,
    random_state        = 42,
    verbose             = 1,
    n_jobs              = -1
)

rf_search.fit(X_train, y_train)

print('Best params:', rf_search.best_params_)
print('Best CV AUC:', rf_search.best_score_)

rf_tuned = rf_search.best_estimator_
```

> If tuning improves AUC by less than 0.5%, stick with the baseline. Simpler is more clinically defensible.

---

## Step 3 — Final Evaluation on Test Set

**The test set is used exactly once — after all tuning is locked.**

```python
from sklearn.metrics import (classification_report, roc_auc_score,
                              brier_score_loss, confusion_matrix)

rf_final = rf_tuned  # or rf_baseline if tuning didn't help
rf_final.fit(X_train, y_train)

y_pred = rf_final.predict(X_test)
y_prob = rf_final.predict_proba(X_test)[:, 1]

print(classification_report(y_test, y_pred, target_names=['Benign', 'Malignant']))
print(f'ROC-AUC : {roc_auc_score(y_test, y_prob):.4f}')
print(f'Brier   : {brier_score_loss(y_test, y_prob):.4f}')
```

### Clinical Confusion Matrix

```python
cm = confusion_matrix(y_test, y_pred)
tn, fp, fn, tp = cm.ravel()

print(f'True Negatives  (Correct Benign)   : {tn}')
print(f'True Positives  (Caught Cancers)   : {tp}')
print(f'False Positives (Unnecessary CNBs) : {fp}')
print(f'False Negatives (MISSED CANCERS)   : {fn}  <-- Most critical')

sensitivity = tp / (tp + fn)
specificity = tn / (tn + fp)
print(f'Sensitivity: {sensitivity:.4f}')
print(f'Specificity: {specificity:.4f}')
```

> **Your presentation headline lives here.** "OncoTriage AI missed only X cancers in 114 evaluated patients."

---

## Step 4 — Probability Calibration

Raw RF probabilities cluster around 0.3–0.7 and are poorly calibrated. This directly corrupts the Reliability Score's certainty component. Fix it with Platt Scaling.

```python
from sklearn.calibration import CalibratedClassifierCV, calibration_curve

rf_calibrated = CalibratedClassifierCV(
    rf_final,
    method = 'sigmoid',   # Platt scaling
    cv     = 5
)
rf_calibrated.fit(X_train, y_train)

# Compare Brier scores
y_prob_raw = rf_final.predict_proba(X_test)[:, 1]
y_prob_cal = rf_calibrated.predict_proba(X_test)[:, 1]

print(f'Brier (uncalibrated): {brier_score_loss(y_test, y_prob_raw):.4f}')
print(f'Brier (calibrated)  : {brier_score_loss(y_test, y_prob_cal):.4f}')
```

> **Use rf_calibrated for ALL downstream work** — Reliability Score, SHAP, Streamlit. Never expose uncalibrated probabilities in a clinical output.

---

## Step 5 — Feature Importance (Pre-SHAP Sanity Check)

```python
import pandas as pd
import matplotlib.pyplot as plt

importances = rf_final.feature_importances_
feat_df = pd.DataFrame({
    'feature'   : features,
    'importance': importances
}).sort_values('importance', ascending=False)

print(feat_df.to_string(index=False))

plt.figure(figsize=(10, 6))
plt.barh(feat_df['feature'], feat_df['importance'], color='steelblue')
plt.xlabel('Mean Decrease in Impurity')
plt.title('RF Feature Importances — Pre-SHAP Sanity Check')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig('outputs/rf_feature_importance.png', dpi=150)
```

> Expected top features based on EDA: `area_worst`, `concave_points_mean`, `smoothness_worst`. If fractal dimension ranks in top 5, investigate — the violin plots suggested it was a weak discriminator.

---

## Step 6 — Export All Artifacts

```python
import joblib, json

# Models
joblib.dump(rf_final,      'outputs/artifacts/rf_model.joblib')
joblib.dump(rf_calibrated, 'outputs/artifacts/rf_calibrated.joblib')

# Probability arrays for ensemble in Phase 3b
np.save('outputs/intermediate/rf_y_prob_train.npy',
         rf_calibrated.predict_proba(X_train)[:, 1])
np.save('outputs/intermediate/rf_y_prob_test.npy', y_prob_cal)

# Performance metrics
rf_metrics = {
    'roc_auc'    : float(roc_auc_score(y_test, y_prob_cal)),
    'brier'      : float(brier_score_loss(y_test, y_prob_cal)),
    'sensitivity': float(sensitivity),
    'specificity': float(specificity),
    'fn_count'   : int(fn),
    'tp_count'   : int(tp)
}
json.dump(rf_metrics, open('outputs/artifacts/rf_metrics.json', 'w'), indent=2)
print('All RF artifacts saved.')
```

### Artifact Reference

| Artifact | Used In |
|---|---|
| `rf_model.joblib` | Phase 4 SHAP (TreeExplainer needs base estimator) |
| `rf_calibrated.joblib` | All clinical outputs, Streamlit demo |
| `rf_y_prob_train.npy` | Phase 3b model agreement calculation |
| `rf_y_prob_test.npy` | Phase 3b model agreement calculation |
| `rf_metrics.json` | Results summary, presentation claims |

---

## Step 7 — Completion Checklist

Before moving to Phase 3b (Logistic Regression + XGBoost):

- [ ] Baseline CV complete — Sensitivity ≥ 0.93, AUC ≥ 0.97
- [ ] RandomizedSearchCV (30 iterations) complete — best params logged
- [ ] Test set evaluated exactly once — metrics in `rf_metrics.json`
- [ ] Calibrated Brier ≤ 0.07
- [ ] Feature importance plot saved to `outputs/`
- [ ] All 5 artifacts saved and load-tested
- [ ] No test data appeared in any `.fit()` call

---

## How RF Feeds Everything Downstream

| Component | Needs from RF |
|---|---|
| SHAP Explainability | `rf_model.joblib` — TreeExplainer requires the base estimator |
| Reliability Score — Certainty | Calibrated probabilities |
| Reliability Score — Agreement | `rf_y_prob_*.npy` for inter-model variance |
| Conformal Prediction | Calibrated probability scores |
| Streamlit Demo | `rf_calibrated.joblib` + `scaler.joblib` |
| Edge Case Simulator | Live probability + reliability score from calibrated model |

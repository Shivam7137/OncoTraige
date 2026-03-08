# OncoTriage AI

A reliability-aware predictive oncology triage system focused on the Breast Cancer Wisconsin (Diagnostic) Dataset (UCI ID: 17). 

This project aims to build an interpretable, robust machine learning pipeline where false negatives (missing a malignancy) are heavily penalized, and prediction boundary cases are escalated via Conformal Prediction.

## Project Structure & Progress

The pipeline is currently divided into modular scripts covering Advanced Exploratory Data Analysis (Phase 1) and robust preprocessing (Phase 2).

### Phase 1: Advanced Exploratory Data Analysis & Topological Mapping
Investigative EDA proving the necessity of a reliability-aware architecture. The legacy monolithic analysis script was refactored into modular components:

*   **`p1_00_config.py`**: Handles UCI dataset fetching, local caching (`data.csv`), canonical feature renaming, and shared visualization constants.
*   **`p1_01_target_analysis.py`**: Analyzes the class distribution (63% Benign, 37% Malignant) and establishes the asymmetric clinical cost matrix (prioritizing Sensitivity/AUROC over standard Accuracy).
*   **`p1_02_skewness_analysis.py`**: Visualizes biological skewness in size and shape features, deciding on a **StandardScaler + Winsorization** strategy to appropriately manage the presence of extreme pathological outliers without compromising clinical transparency.
*   **`p1_03_pca_analysis.py`**: Performs 2D PCA projection, highlighting the chaotic decision boundary and justifying the future use of Conformal Prediction for edge cases.
*   **`p1_04_vif_analysis.py`**: Computes Variance Inflation Factors (VIF) across all 30 features and implements iterative pruning to eliminate severe multicollinearity (saving the 17 surviving features to `surviving_features.json`).
*   **`p1_05_post_vif_analysis.py`**: Re-evaluates target separability using only the 17 VIF-surviving features to ensure predictive signal was not lost.

All Phase 1 visualizations are saved to the `outputs/` directory.

### Phase 2: Preprocessing Pipeline
Strict implementation of data preprocessing rules designed to prevent data leakage and handle clinical realities:

*   **`p2_01_split_and_select.py`**: Loads the cached data, drops leakage identifiers, applies 80/20 Stratified train/test splitting, and reduces the dataset to the 17 VIF-surviving features. Intermediates are saved to `outputs/intermediate/`.
*   **`p2_02_scale_and_winsorize.py`**: Scales features using `StandardScaler` (fit **strictly** on the training set) and caps extreme outliers using 1st/99th percentile Winsorization (bounds calculated **strictly** on the training set). 
*   **`p2_03_class_weights.py`**: Computes robust `balanced` class weights to combat the 63:37 benign/malignant disparity, avoiding synthetic data generation strategies (like SMOTE) that distort clinical purity.

#### Phase 2 Artifacts Exported
Prepared for the final application deployment in `outputs/artifacts/`:
*   `scaler.joblib`: The fitted StandardScaler.
*   `winsorize_bounds.json`: The 1st and 99th percentile boundaries used for capping new patient inputs.
*   `class_weights.json`: The balanced class weights for use during model instantiation.

## Installation & Usage

1. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
2. **Run Phase 1 (EDA & Visualization):**
   ```bash
   python p1_01_target_analysis.py
   python p1_02_skewness_analysis.py
   python p1_03_pca_analysis.py
   python p1_04_vif_analysis.py
   python p1_05_post_vif_analysis.py
   ```
3. **Run Phase 2 (Preprocessing):**
   *(Must run Phase 1 first to generate `surviving_features.json`)*
   ```bash
   python p2_01_split_and_select.py
   python p2_02_scale_and_winsorize.py
   python p2_03_class_weights.py
   ```

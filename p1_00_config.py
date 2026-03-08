"""
Shared configuration, styling, and data loading for OncoTriage AI - Phase 1.
"""
import os
import sys
import warnings
import pandas as pd
from ucimlrepo import fetch_ucirepo
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

# Force UTF-8 output on Windows
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

# ─── Configuration ─────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)
DATA_PATH = os.path.join(BASE_DIR, "data.csv")

# Color palette — clinical styling
BENIGN_COLOR = "#2196F3"
MALIGNANT_COLOR = "#E53935"
BG_COLOR = "#0D1117"
PANEL_COLOR = "#161B22"
TEXT_COLOR = "#E6EDF3"
GRID_COLOR = "#21262D"
ACCENT_CYAN = "#58A6FF"

plt.rcParams.update({
    "figure.facecolor": BG_COLOR,
    "axes.facecolor": PANEL_COLOR,
    "axes.edgecolor": GRID_COLOR,
    "axes.labelcolor": TEXT_COLOR,
    "text.color": TEXT_COLOR,
    "xtick.color": TEXT_COLOR,
    "ytick.color": TEXT_COLOR,
    "grid.color": GRID_COLOR,
    "grid.alpha": 0.3,
    "font.family": "sans-serif",
    "font.size": 11,
    "axes.titlesize": 14,
    "axes.labelsize": 12,
    "figure.titlesize": 18,
})


def section_header(title, section_num):
    width = 76
    print("\n" + "═" * width)
    print(f"  SECTION {section_num}: {title}")
    print("═" * width)


def save_figure(fig, filename):
    path = os.path.join(OUTPUT_DIR, filename)
    fig.savefig(path, dpi=200, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  ✓ Saved: {filename}")


def load_dataset():
    """Loads dataset from UCI or local cache, cleans columns, returns X, y, y_labels."""
    if not os.path.exists(DATA_PATH):
        print("  Fetching Breast Cancer Wisconsin (Diagnostic) dataset from UCI...")
        dataset = fetch_ucirepo(id=17)
        # Create full dataframe and save to csv
        df_full = pd.concat([dataset.data.targets, dataset.data.features], axis=1)
        
        # Rename columns to canonical names before saving
        base_features = [
            "radius", "texture", "perimeter", "area", "smoothness",
            "compactness", "concavity", "concave_points", "symmetry",
            "fractal_dimension"
        ]
        rename_map = {}
        for feat in base_features:
            rename_map[f"{feat}1"] = f"{feat}_mean"
            rename_map[f"{feat}2"] = f"{feat}_se"
            rename_map[f"{feat}3"] = f"{feat}_worst"
        
        df_full.rename(columns=rename_map, inplace=True)
        # Handle the one space in concave_points from UCIML if it exists
        df_full.rename(columns={'concave points1': 'concave_points_mean',
                                'concave points2': 'concave_points_se',
                                'concave points3': 'concave_points_worst'}, inplace=True)
                                
        df_full.to_csv(DATA_PATH, index=False)
        print("  Cached dataset to data.csv")
    
    # Load from cache
    df = pd.read_csv(DATA_PATH).drop(columns=['id', 'Unnamed: 32'], errors='ignore')
    
    # Standardize 'Diagnosis' column name if it changed
    diag_col = 'Diagnosis' if 'Diagnosis' in df.columns else 'diagnosis'
    
    y = df[[diag_col]]
    X = df.drop(columns=[diag_col])
    
    y_series = y[diag_col].copy()
    label_map = {"M": "Malignant", "B": "Benign"}
    y_labels = y_series.map(label_map)
    
    return X, y, y_labels

# Load once when module is imported
X, y, y_labels = load_dataset()

# ─── Pipeline / Model Settings (from rebuild r0–r7) ─────────────────
ARTIFACT_DIR = os.path.join(OUTPUT_DIR, 'artifacts')
INTER_DIR    = os.path.join(OUTPUT_DIR, 'intermediate')
PLOT_DIR     = os.path.join(OUTPUT_DIR, 'plots')
METRIC_DIR   = os.path.join(OUTPUT_DIR, 'metrics')

for d in [ARTIFACT_DIR, INTER_DIR, PLOT_DIR, METRIC_DIR]:
    os.makedirs(d, exist_ok=True)

# Model settings (confirmed by rebuild ablation)
RANDOM_STATE  = 42
TEST_SIZE     = 0.20
THRESHOLD     = 0.20            # user updated from 0.25
CLASS_WEIGHT  = {0: 1, 1: 4}   # from r3
N_ESTIMATORS  = 500
MAX_FEATURES  = 'sqrt'

# Feature settings — new pipeline uses all 30 features, no VIF reduction
DROP_COLS     = ['id', 'Unnamed: 32']
TARGET_COL    = 'Diagnosis'

# Reliability Score weights (from r6)
REL_WEIGHT_MARGIN    = 0.50
REL_WEIGHT_SHAP      = 0.30
REL_WEIGHT_LEAF      = 0.20
REL_HIGH_THRESHOLD   = 0.80
REL_LOW_THRESHOLD    = 0.50

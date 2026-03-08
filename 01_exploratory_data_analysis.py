# --- content from p1_00_config.py ---
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
THRESHOLD     = 0.25            # from r2
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



# --- content from p1_01_target_analysis.py ---
"""
Phase 1 - Section 1: Target Variable & Asymmetric Risk Analysis
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap

from p1_00_config import (
    X, y, y_labels,
    BENIGN_COLOR, MALIGNANT_COLOR, PANEL_COLOR, TEXT_COLOR,
    section_header, save_figure
)

section_header("TARGET VARIABLE & ASYMMETRIC RISK ANALYSIS", 1)

# 1. Compute class distribution
class_counts = y_labels.value_counts()
class_pct = y_labels.value_counts(normalize=True) * 100

print(f"  Benign instances:    {class_counts['Benign']} ({class_pct['Benign']:.1f}%)")
print(f"  Malignant instances: {class_counts['Malignant']} ({class_pct['Malignant']:.1f}%)")

# 2. Bar chart
fig, ax = plt.subplots(figsize=(8, 5))
bars = ax.bar(["Benign", "Malignant"], class_counts.values, 
              color=[BENIGN_COLOR, MALIGNANT_COLOR], 
              edgecolor='white', linewidth=1.5, alpha=0.9)

for bar, pct in zip(bars, class_pct.values):
    height = bar.get_height()
    ax.annotate(f"{height}\n({pct:.1f}%)",
                xy=(bar.get_x() + bar.get_width() / 2, height / 2),
                xytext=(0, 0), textcoords="offset points",
                ha="center", va="center", color="white",
                fontsize=14, fontweight="bold")

ax.set_title("Breast Cancer Wisconsin: Target Distribution", pad=20, fontweight="bold")
ax.set_ylabel("Number of Patients", labelpad=10)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
save_figure(fig, "01_target_distribution.png")

# 3. Clinical Cost Matrix Heatmap
cost_matrix = np.array([[0, 2], [5, 0]])
fig2, ax2 = plt.subplots(figsize=(6, 5))

# Create a custom colormap from dark to the ACCENT_CYAN/MALIGNANT_COLOR
cmap = LinearSegmentedColormap.from_list("clinical_risk", [PANEL_COLOR, MALIGNANT_COLOR])
sns.heatmap(cost_matrix, annot=True, cmap=cmap, cbar=False,
            xticklabels=["Predicted Benign", "Predicted Malignant"],
            yticklabels=["Actual Benign", "Actual Malignant"],
            fmt="d", annot_kws={"size": 24, "weight": "bold"},
            linewidths=2, linecolor=PANEL_COLOR, ax=ax2)

# Annotate False Positive / False Negative
ax2.text(0.5, 0.2, "True Negative\n(0 Cost)", ha="center", va="center", color=TEXT_COLOR, fontsize=10)
ax2.text(1.5, 0.2, "False Positive\n(Anxiety, Biopsy)", ha="center", va="center", color="white", fontsize=10)
ax2.text(0.5, 1.2, "False Negative\n(FATAL OMISSION)", ha="center", va="center", color="white", fontsize=10, fontweight="bold")
ax2.text(1.5, 1.2, "True Positive\n(0 Cost)", ha="center", va="center", color=TEXT_COLOR, fontsize=10)

ax2.set_title("Clinical Asymmetric Risk Context", pad=20, fontweight="bold")
save_figure(fig2, "02_cost_matrix.png")

print("\n  [ANALYSIS CONCLUSION]")
print("  The dataset presents a 63:37 ratio. While not highly imbalanced mathematically,")
print("  the deeply asymmetric clinical cost dictates that Accuracy is an invalid metric.")
print("  We must optimize for Sensitivity (Recall) and AUROC.")



# --- content from p1_02_skewness_analysis.py ---
"""
Phase 1 - Section 2: Biological Skewness & Pathological Outlier Preservation
"""
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from p1_00_config import (
    X, y_labels,
    BENIGN_COLOR, MALIGNANT_COLOR, 
    section_header, save_figure
)

section_header("BIOLOGICAL SKEWNESS & PATHOLOGICAL OUTLIER PRESERVATION", 2)

df = X.copy()
df["Diagnosis"] = y_labels.values

# --- 2a: KDE plots for size-dependent features ---
size_features = ["area_worst", "radius_worst", "perimeter_worst", "area_mean"]

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle("ANALYSIS 2a — KDE: Size-Dependent Morphological Features\n"
             "Benign (homeostasis) vs. Malignant (mitotic proliferation)",
             fontsize=14, fontweight="bold", y=1.02)

for ax, feat in zip(axes.flatten(), size_features):
    for diag, color, label in [("Benign", BENIGN_COLOR, "Benign"),
                                ("Malignant", MALIGNANT_COLOR, "Malignant")]:
        subset = df[df["Diagnosis"] == diag][feat]
        skew_val = subset.skew()
        sns.kdeplot(subset, ax=ax, color=color, fill=True, alpha=0.3,
                    linewidth=2, label=f"{label} (skew={skew_val:.2f})")

    ax.set_title(feat.replace("_", " ").title(), fontweight="bold")
    ax.set_xlabel(feat)
    ax.set_ylabel("Density")
    ax.legend(fontsize=9, framealpha=0.3)
    ax.grid(alpha=0.15)

fig.tight_layout()
save_figure(fig, "03_kde_size_features.png")

# --- 2b: Violin plots for geometric complexity features ---
complexity_features = ["concavity_worst", "compactness_worst",
                       "concave_points_worst", "fractal_dimension_worst"]

# Correct missing column names if dataset uses spaces
col_mapper = {col: col.replace(' ', '_') for col in df.columns}
df = df.rename(columns=col_mapper)

fig2, axes2 = plt.subplots(2, 2, figsize=(14, 10))
fig2.suptitle("ANALYSIS 2b — Violin Plots: Geometric Complexity Features\n"
             "Tight benign distributions vs. long-tailed malignant outliers",
             fontsize=14, fontweight="bold", y=1.02)

palette = {"Benign": BENIGN_COLOR, "Malignant": MALIGNANT_COLOR}

for ax, feat in zip(axes2.flatten(), complexity_features):
    if feat in df.columns:
        sns.violinplot(data=df, x="Diagnosis", y=feat, ax=ax,
                    palette=palette, inner="quartile", linewidth=1.2,
                    order=["Benign", "Malignant"])
        ax.set_title(feat.replace("_", " ").title(), fontweight="bold")
        ax.grid(axis="y", alpha=0.15)
    else:
        print(f"Warning: Feature {feat} not found.")

fig2.tight_layout()
save_figure(fig2, "04_violin_complexity_features.png")

# --- 2c: Print skewness statistics ---
print("\n  Skewness Statistics (per class):\n")
print(f"  {'Feature':<30s}  {'Benign':>10s}  {'Malignant':>10s}  {'Δ Skew':>10s}")
print("  " + "─" * 66)

all_skew_features = size_features + complexity_features
for feat in all_skew_features:
    if feat in df.columns:
        sk_b = df[df["Diagnosis"] == "Benign"][feat].skew()
        sk_m = df[df["Diagnosis"] == "Malignant"][feat].skew()
        print(f"  {feat:<30s}  {sk_b:>10.3f}  {sk_m:>10.3f}  {abs(sk_m - sk_b):>10.3f}")

print("\n  ┌──────────────────────────────────────────────────────────────────┐")
print("  │  ARCHITECTURAL DECISION: Scaling Strategy                       │")
print("  ├──────────────────────────────────────────────────────────────────┤")
print("  │  ✗ RobustScaler REJECTED — while outlier-resistant, its        │")
print("  │    median/IQR logic can be harder to explain clinically.       │")
print("  │                                                                │")
print("  │  ✓ StandardScaler + Winsorization ADOPTED — standardizes to   │")
print("  │    mean 0, std 1. We compensate for its outlier sensitivity    │")
print("  │    by applying 1st/99th percentile Winsorization (capping).    │")
print("  │    This protects the model from distortion while maintaining   │")
print("  │    a highly transparent and explainable two-step workflow.     │")
print("  └──────────────────────────────────────────────────────────────────┘")



# --- content from p1_03_pca_analysis.py ---
"""
Phase 1 - Section 3: Topological Separability & The Triage Boundary
"""
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

from p1_00_config import (
    X, y_labels,
    BENIGN_COLOR, MALIGNANT_COLOR, BG_COLOR, PANEL_COLOR, GRID_COLOR,
    section_header, save_figure
)

section_header("TOPOLOGICAL SEPARABILITY & THE TRIAGE BOUNDARY", 3)

# --- 3a: PCA projection ---
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

print(f"\n  Explained variance ratio:")
print(f"    PC1: {pca.explained_variance_ratio_[0]:.4f} "
      f"({pca.explained_variance_ratio_[0]*100:.1f}%)")
print(f"    PC2: {pca.explained_variance_ratio_[1]:.4f} "
      f"({pca.explained_variance_ratio_[1]*100:.1f}%)")
print(f"    Total: {sum(pca.explained_variance_ratio_)*100:.1f}%")

# --- 3b: 2D scatter with boundary zone ---
pca_df = pd.DataFrame(X_pca, columns=["PC1", "PC2"])
pca_df["Diagnosis"] = y_labels.values

fig, ax = plt.subplots(figsize=(12, 9))
fig.suptitle("ANALYSIS 3 — PCA Topological Separability Map\n"
             "30D Feature Space → 2D Principal Component Projection",
             fontsize=14, fontweight="bold", y=0.98)

# Plot each class
for diag, color, marker, zorder in [
    ("Benign", BENIGN_COLOR, "o", 2),
    ("Malignant", MALIGNANT_COLOR, "^", 3)
]:
    subset = pca_df[pca_df["Diagnosis"] == diag]
    ax.scatter(subset["PC1"], subset["PC2"],
               c=color, marker=marker, s=45, alpha=0.65,
               edgecolors="white", linewidths=0.4,
               label=f"{diag} (n={len(subset)})", zorder=zorder)

benign_pc1 = pca_df[pca_df["Diagnosis"] == "Benign"]["PC1"]
malig_pc1 = pca_df[pca_df["Diagnosis"] == "Malignant"]["PC1"]

overlap_left = max(benign_pc1.min(), malig_pc1.quantile(0.05))
overlap_right = min(benign_pc1.quantile(0.95), malig_pc1.max())

ax.axvspan(overlap_left, overlap_right, alpha=0.08, color="#FFAB00",
           zorder=1, label="Triage Boundary Zone")
ax.axvline(overlap_left, color="#FFAB00", linestyle="--", alpha=0.5,
           linewidth=1, zorder=1)
ax.axvline(overlap_right, color="#FFAB00", linestyle="--", alpha=0.5,
           linewidth=1, zorder=1)

ax.annotate("⚠ CHAOTIC OVERLAP\nConformal Prediction\nEscalation Zone",
            xy=((overlap_left + overlap_right) / 2,
                ax.get_ylim()[1] * 0.85),
            fontsize=10, fontweight="bold", color="#FFAB00",
            ha="center", va="top",
            bbox=dict(boxstyle="round,pad=0.5", facecolor=BG_COLOR,
                      edgecolor="#FFAB00", alpha=0.9))

ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% variance)", fontsize=12)
ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% variance)", fontsize=12)
ax.legend(fontsize=11, loc="lower right", framealpha=0.7,
          facecolor=PANEL_COLOR, edgecolor=GRID_COLOR)
ax.grid(alpha=0.15)

save_figure(fig, "05_pca_scatter.png")

n_boundary = len(pca_df[(pca_df["PC1"] >= overlap_left) &
                         (pca_df["PC1"] <= overlap_right)])
print(f"\n  Samples in the boundary zone: {n_boundary} / {len(pca_df)} "
      f"({n_boundary/len(pca_df)*100:.1f}%)")

print("\n  ┌──────────────────────────────────────────────────────────────────┐")
print("  │  ARCHITECTURAL DECISION: Conformal Prediction Escalation          │")
print("  ├──────────────────────────────────────────────────────────────-────┤")
print("  │  The PCA map reveals a chaotic frontier where benign and        │")
print("  │  malignant clusters intermingle. Deterministic classification   │")
print("  │  in this zone is clinically dangerous.                          │")
print("  │                                                                 │")
print("  │  ✓ Conformal Prediction ADOPTED — when a patient's feature      │")
print("  │    vector falls into the boundary zone, the system assigns      │")
print("  │    LOW confidence and triggers automatic escalation to a        │")
print("  │    Core Needle Biopsy (CNB) rather than forcing a guess.        │")
print("  └──────────────────────────────────────────────────────────────────┘")



# --- content from p1_04_vif_analysis.py ---
"""
Phase 1 - Section 4: High-Dimensional Multicollinearity Deconstruction
"""
import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.outliers_influence import variance_inflation_factor

from p1_00_config import (
    X, BENIGN_COLOR, MALIGNANT_COLOR, BG_COLOR, BASE_DIR,
    section_header, save_figure
)

section_header("HIGH-DIMENSIONAL MULTICOLLINEARITY DECONSTRUCTION", 4)

corr_matrix = X.corr()

fig, ax = plt.subplots(figsize=(18, 15))
fig.suptitle("ANALYSIS 4a — Pairwise Pearson Correlation Matrix (30 Features)\n"
             "Exposing severe multicollinearity in the morphological feature space",
             fontsize=14, fontweight="bold", y=0.98)

cmap_corr = LinearSegmentedColormap.from_list(
    "corr", [BENIGN_COLOR, "#1A1E24", MALIGNANT_COLOR], N=256
)
mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
sns.heatmap(corr_matrix, mask=mask, ax=ax, cmap=cmap_corr,
            center=0, vmin=-1, vmax=1,
            square=True, linewidths=0.5, linecolor=BG_COLOR,
            annot=False,
            cbar_kws={"shrink": 0.7, "label": "Pearson r"})
ax.tick_params(axis="x", rotation=90, labelsize=8)
ax.tick_params(axis="y", rotation=0, labelsize=8)

save_figure(fig, "06_correlation_heatmap.png")

# --- 4b: Identify extreme correlations ---
print("\n  Feature Pairs with |r| > 0.95:\n")
extreme_pairs = []
for i in range(len(corr_matrix.columns)):
    for j in range(i + 1, len(corr_matrix.columns)):
        r = corr_matrix.iloc[i, j]
        if abs(r) > 0.95:
            f1 = corr_matrix.columns[i]
            f2 = corr_matrix.columns[j]
            extreme_pairs.append((f1, f2, r))

print(f"\n  Total pairs with |r| > 0.95: {len(extreme_pairs)}")

# --- 4c: VIF calculation ---
print("\n  Computing Variance Inflation Factors (VIF) for all 30 features...")
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_vif = pd.DataFrame(X_scaled, columns=X.columns)

vif_data = pd.DataFrame({
    "Feature": X_vif.columns,
    "VIF": [variance_inflation_factor(X_vif.values, i) for i in range(X_vif.shape[1])]
})
vif_data = vif_data.sort_values("VIF", ascending=False).reset_index(drop=True)

# --- 4d: Iterative VIF reduction ---
print("\n\n  ── Iterative VIF Reduction Algorithm ──")
print("  Threshold: VIF < 10.0\n")

X_reduced = X_vif.copy()
dropped_features = []
iteration = 0

while True:
    iteration += 1
    vifs = pd.Series(
        [variance_inflation_factor(X_reduced.values, i) for i in range(X_reduced.shape[1])],
        index=X_reduced.columns
    )
    max_vif = vifs.max()
    max_feat = vifs.idxmax()

    if max_vif < 10.0:
        break

    dropped_features.append((max_feat, max_vif))
    X_reduced = X_reduced.drop(columns=[max_feat])

surviving_features = list(X_reduced.columns)

print(f"\n  Features dropped:   {len(dropped_features)}")
print(f"  Features retained:  {len(surviving_features)}\n")

# Save surviving features to file for later phases
with open(os.path.join(BASE_DIR, "surviving_features.json"), "w") as f:
    json.dump(surviving_features, f)
print("  ✓ Saved: surviving_features.json")

# --- 4e: VIF comparison visualization ---
fig2, axes2 = plt.subplots(1, 2, figsize=(16, 8))
fig2.suptitle("ANALYSIS 4b — VIF Reduction: Before vs. After\n"
             "Iterative removal of redundant features (threshold: VIF < 10)",
             fontsize=14, fontweight="bold", y=1.02)

ax = axes2[0]
top_n = min(15, len(vif_data))
top_vif = vif_data.head(top_n)
colors_vif = [MALIGNANT_COLOR if v > 100 else "#FF9800" if v > 10 else BENIGN_COLOR for v in top_vif["VIF"]]
ax.barh(range(top_n), top_vif["VIF"].values, color=colors_vif, edgecolor="white", linewidth=0.5, height=0.7)
ax.set_yticks(range(top_n))
ax.set_yticklabels(top_vif["Feature"].values, fontsize=8)
ax.axvline(10, color="#FFAB00", linestyle="--", linewidth=1.5)
ax.set_xlabel("VIF")
ax.set_title(f"BEFORE ({len(vif_data)} features)", fontweight="bold")
ax.invert_yaxis()

ax2 = axes2[1]
vif_after = pd.Series(
    [variance_inflation_factor(X_reduced.values, i) for i in range(X_reduced.shape[1])],
    index=X_reduced.columns
).sort_values(ascending=False)
colors_after = [BENIGN_COLOR if v < 10 else "#FF9800" for v in vif_after.values]
ax2.barh(range(len(vif_after)), vif_after.values, color=colors_after, edgecolor="white", linewidth=0.5, height=0.7)
ax2.set_yticks(range(len(vif_after)))
ax2.set_yticklabels(vif_after.index, fontsize=8)
ax2.axvline(10, color="#FFAB00", linestyle="--", linewidth=1.5)
ax2.set_xlabel("VIF")
ax2.set_title(f"AFTER ({len(surviving_features)} features)", fontweight="bold")
ax2.invert_yaxis()

fig2.tight_layout()
save_figure(fig2, "07_vif_analysis.png")



# --- content from p1_05_post_vif_analysis.py ---
"""
Phase 1 - Section 5: Post-VIF Topological Comparison
"""
import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.decomposition import PCA

from p1_00_config import (
    X, y_labels, BENIGN_COLOR, MALIGNANT_COLOR, BG_COLOR, PANEL_COLOR, GRID_COLOR, BASE_DIR,
    section_header, save_figure
)

section_header("POST-VIF TOPOLOGICAL COMPARISON", 5)

# Load surviving features
features_path = os.path.join(BASE_DIR, "surviving_features.json")
if not os.path.exists(features_path):
    print("Error: surviving_features.json not found. Run p1_04_vif_analysis.py first.")
    exit(1)

with open(features_path, "r") as f:
    surviving_features = json.load(f)

X_reduced = X[surviving_features]

# --- 5a: Side-by-side correlation heatmap ---
corr_before = X.corr()
corr_after = X_reduced.corr()

fig, axes = plt.subplots(1, 2, figsize=(24, 10))
fig.suptitle("ANALYSIS 5a — Correlation Heatmap: Before vs. After VIF Reduction\n"
             f"30 features (severe multicollinearity) vs. {len(surviving_features)} "
             "VIF-surviving features (cleaned)",
             fontsize=14, fontweight="bold", y=1.02)

cmap_corr = LinearSegmentedColormap.from_list(
    "corr", [BENIGN_COLOR, "#1A1E24", MALIGNANT_COLOR], N=256
)

# Left — Before
ax1 = axes[0]
mask1 = np.triu(np.ones_like(corr_before, dtype=bool), k=1)
sns.heatmap(corr_before, mask=mask1, ax=ax1, cmap=cmap_corr,
            center=0, vmin=-1, vmax=1, square=True, linewidths=0.3, linecolor=BG_COLOR,
            annot=False, cbar_kws={"shrink": 0.6})
ax1.set_title("BEFORE VIF Reduction", fontweight="bold", fontsize=12)

# Right — After
ax2 = axes[1]
mask2 = np.triu(np.ones_like(corr_after, dtype=bool), k=1)
sns.heatmap(corr_after, mask=mask2, ax=ax2, cmap=cmap_corr,
            center=0, vmin=-1, vmax=1, square=True, linewidths=0.3, linecolor=BG_COLOR,
            annot=True, fmt=".2f", annot_kws={"size": 6}, cbar_kws={"shrink": 0.6})
ax2.set_title(f"AFTER VIF Reduction ({len(surviving_features)} features)", fontweight="bold", fontsize=12)

fig.tight_layout()
save_figure(fig, "09_correlation_post_vif.png")

# --- 5b: PCA on VIF-reduced features ---
pca_orig = PCA(n_components=2)
X_pca_orig = pca_orig.fit_transform(StandardScaler().fit_transform(X))
pca_df = pd.DataFrame(X_pca_orig, columns=["PC1", "PC2"])
pca_df["Diagnosis"] = y_labels.values

scaler_post = RobustScaler()
X_reduced_scaled = scaler_post.fit_transform(X_reduced)
pca_post = PCA(n_components=2)
pca_post_result = pca_post.fit_transform(X_reduced_scaled)
pca_post_df = pd.DataFrame(pca_post_result, columns=["PC1", "PC2"])
pca_post_df["Diagnosis"] = y_labels.values

fig2, axes2 = plt.subplots(1, 2, figsize=(20, 8))
fig2.suptitle("ANALYSIS 5 - PCA Topological Map: Before vs After VIF", fontsize=14, fontweight="bold", y=1.02)

ax1 = axes2[0]
for diag, color, marker in [("Benign", BENIGN_COLOR, "o"), ("Malignant", MALIGNANT_COLOR, "X")]:
    subset = pca_df[pca_df["Diagnosis"] == diag]
    ax1.scatter(subset["PC1"], subset["PC2"], c=color, marker=marker, alpha=0.6, edgecolors="white", linewidth=0.3, s=40, label=diag)
ax1.set_title(f"BEFORE VIF (30 features)\nTotal explained: {sum(pca_orig.explained_variance_ratio_)*100:.1f}%")

ax2 = axes2[1]
for diag, color, marker in [("Benign", BENIGN_COLOR, "o"), ("Malignant", MALIGNANT_COLOR, "X")]:
    subset = pca_post_df[pca_post_df["Diagnosis"] == diag]
    ax2.scatter(subset["PC1"], subset["PC2"], c=color, marker=marker, alpha=0.6, edgecolors="white", linewidth=0.3, s=40, label=diag)
ax2.set_title(f"AFTER VIF ({len(surviving_features)} features)\nTotal explained: {sum(pca_post.explained_variance_ratio_)*100:.1f}%")

save_figure(fig2, "08_pca_post_vif.png")

print("\n  Summary: Phase 1 Modular EDA Complete.")




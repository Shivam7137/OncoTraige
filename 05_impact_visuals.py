"""
r7_visuals.py — Preprocessing Impact Plot (Live Runs)
=========================================================
Runs exactly four stages on the dataset to evaluate how the original 
preprocessing steps (VIF) compared to the final pipeline (Thresholding & 
Class Weights), and outputs a two-panel visualization.
"""

import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, roc_auc_score, brier_score_loss

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def main():
    print("=" * 60)
    print("  r7 — PREPROCESSING IMPACT VISUALIZATION (Live Runs)")
    print("=" * 60)
    
    # ── 1. Load Data & Stratified Split ──
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(BASE_DIR, 'data.csv')
    
    if not os.path.exists(data_path):
        print(f"Error: {data_path} not found.")
        return
        
    df = pd.read_csv(data_path)
    
    if 'Unnamed: 32' in df.columns:
        df = df.drop(columns=['Unnamed: 32'])
        
    df['Diagnosis'] = (df['Diagnosis'] == 'M').astype(int)
    y = df['Diagnosis']
    X = df.drop(columns=['Diagnosis'])
    
    # 80/20 Stratified split with fixed random state
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    
    # Hardcoded 17 VIF features from identical earlier p1 phase
    vif_features = [
        "texture_mean", "smoothness_mean", "concave_points_mean", "symmetry_mean", 
        "fractal_dimension_mean", "texture_se", "perimeter_se", "smoothness_se", 
        "compactness_se", "concavity_se", "concave_points_se", "symmetry_se", 
        "fractal_dimension_se", "area_worst", "smoothness_worst", "symmetry_worst", 
        "fractal_dimension_worst"
    ]
    raw_features = list(X.columns)
    
    # ── 2. Stage Configurations ──
    stages_config = [
        {
            'name': 'Stage 1 (Baseline)',
            'display': 'Stage 1\nBaseline\n(Raw Data)',
            'features_list': raw_features,
            'features_count': 30,
            'class_weight': 'balanced',
            'threshold': 0.50
        },
        {
            'name': 'Stage 2 (VIF)',
            'display': 'Stage 2\nVIF\nReduction',
            'features_list': vif_features,
            'features_count': 17,
            'class_weight': 'balanced',
            'threshold': 0.50
        },
        {
            'name': 'Stage 3 (Threshold)',
            'display': 'Stage 3\nThreshold\nTuning',
            'features_list': raw_features, # Returns to 30 raw
            'features_count': 30,
            'class_weight': 'balanced',
            'threshold': 0.25
        },
        {
            'name': 'Stage 4 (Weights)',
            'display': 'Stage 4\nClass Weights\n(Final)',
            'features_list': raw_features,
            'features_count': 30,
            'class_weight': {0: 1, 1: 4},
            'threshold': 0.25
        }
    ]
    
    results = []
    
    # ── 3. Execute 4 Live Runs ──
    print("\n  Executing 4 live stage runs on the identical test set:")
    for config in stages_config:
        print(f"    -> Running {config['name']}...")
        X_tr = X_train[config['features_list']]
        X_te = X_test[config['features_list']]
        
        # Match identical hyperparams from original script
        rf = RandomForestClassifier(n_estimators=500, max_features='sqrt', 
                                    class_weight=config['class_weight'], random_state=42)
        rf.fit(X_tr, y_train)
        
        y_prob = rf.predict_proba(X_te)[:, 1]
        y_pred = (y_prob >= config['threshold']).astype(int)
        
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
        sensitivity = tp / (tp + fn)
        specificity = tn / (tn + fp)
        auc = roc_auc_score(y_test, y_prob)
        brier = brier_score_loss(y_test, y_prob)
        
        results.append({
            'name': config['display'],
            'features': config['features_count'],
            'sens': sensitivity,
            'spec': specificity,
            'auc': auc,
            'brier': brier,
            'rejected': 'VIF' in config['name']
        })
        
    # ── 4. Build the Figure ──
    # Two-panel layout: line chart on top, table on bottom
    fig, (ax_chart, ax_table) = plt.subplots(2, 1, figsize=(11, 9), gridspec_kw={'height_ratios': [2.2, 1]})
    
    # Top Panel: Line Chart
    x_labels = [r['name'] for r in results]
    x_pos = np.arange(len(results))
    
    sens_vals = [r['sens'] for r in results]
    auc_vals = [r['auc'] for r in results]
    spec_vals = [r['spec'] for r in results]
    brier_vals = [r['brier'] for r in results]
    
    ax_chart.plot(x_pos, sens_vals, 'o-', color='#e74c3c', linewidth=2.5, markersize=8, label='Sensitivity (Recall)', zorder=5)
    ax_chart.plot(x_pos, auc_vals, 's-', color='#3498db', linewidth=2.5, markersize=8, label='AUC', zorder=5)
    ax_chart.plot(x_pos, spec_vals, 'D-', color='#2ecc71', linewidth=2.5, markersize=8, label='Specificity', zorder=5)
    ax_chart.plot(x_pos, brier_vals, '^-', color='#f39c12', linewidth=2, markersize=7, label='Brier Score (lower = better)', zorder=5)
    
    for i, r in enumerate(results):
        if r['rejected']:
            ax_chart.axvspan(i - 0.3, i + 0.3, alpha=0.1, color='red')
            ax_chart.text(i, max(sens_vals[i], auc_vals[i]) + 0.04, 'REJECTED\n(Drops Nuance)', 
                          ha='center', fontsize=10, color='#e74c3c', fontweight='bold')
                          
    ax_chart.set_xticks(x_pos)
    ax_chart.set_xticklabels(x_labels, fontsize=10)
    ax_chart.set_ylabel('Score', fontsize=12)
    ax_chart.set_title('Preprocessing Impact on Random Forest Performance', fontsize=14, fontweight='bold', pad=15)
    ax_chart.legend(fontsize=10, loc='center right')
    ax_chart.grid(alpha=0.3)
    ax_chart.set_ylim(0.0, 1.05)
    
    # Bottom Panel: Table
    ax_table.axis('tight')
    ax_table.axis('off')
    
    table_data = []
    for r in results:
        status = "Rejected" if r['rejected'] else "Accepted"
        table_data.append([
            r['name'], 
            f"{r['features']}",
            f"{r['sens']:.3f}", 
            f"{r['spec']:.3f}", 
            f"{r['auc']:.4f}", 
            f"{r['brier']:.4f}",
            status
        ])
        
    col_labels = ["Stage", "Features", "Sensitivity", "Specificity", "AUC", "Brier Loss", "Status"]
    
    # Use bbox to fit the table exactly into the subplot area
    table = ax_table.table(cellText=table_data, colLabels=col_labels, cellLoc='center', loc='center', bbox=[0, 0, 1, 1])
    
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    
    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_text_props(weight='bold', color='white')
            cell.set_facecolor('#2c3e50')
        elif col == 6:
            text = cell.get_text().get_text()
            if 'Rejected' in text:
                cell.set_text_props(color='#e74c3c', weight='bold')
            else:
                cell.set_text_props(color='#27ae60', weight='bold')
                
    plt.tight_layout()
    os.makedirs('outputs/plots', exist_ok=True)
    out_path = os.path.join(BASE_DIR, 'outputs', 'plots', 'preprocessing_impact.png')
    plt.savefig(out_path, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print("\n  Summary Table:")
    print(f"  {'Stage':<25} {'Feats':>5} {'Sens':>8} {'Spec':>8} {'AUC':>8} {'Brier':>8}")
    print("  " + "-"*65)
    for r in results:
        print(f"  {r['name'].replace(chr(10), ' '):<25} {r['features']:>5} {r['sens']:>8.3f} {r['spec']:>8.3f} {r['auc']:>8.4f} {r['brier']:>8.4f}")
        
    print(f"\n  ✓ Saved to: {out_path}")

if __name__ == '__main__':
    main()

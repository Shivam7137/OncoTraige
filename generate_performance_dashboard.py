import os
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
from p1_00_config import (
    INTER_DIR, OUTPUT_DIR, THRESHOLD, BENIGN_COLOR, MALIGNANT_COLOR,
    BG_COLOR, PANEL_COLOR, TEXT_COLOR, GRID_COLOR, ACCENT_CYAN,
    save_figure, section_header
)

def generate_dashboard():
    print("Generating Standalone Confusion Matrix...")
    
    # Load Data
    with open('outputs/artifacts/rf_metrics.json', 'r') as f:
        metrics = json.load(f)
        
    # Set up figure
    fig = plt.figure(figsize=(10, 8), facecolor=BG_COLOR)
    fig.suptitle('OncoSmart AI — Final Model Confusion Matrix', 
                 fontsize=20, fontweight='bold', color='white', y=0.98)
    
    ax_cm = fig.add_subplot(111)
    ax_cm.set_facecolor(PANEL_COLOR)
    
    threshold = metrics['threshold']
    
    cm_data = np.array([[metrics['tn'], metrics['fp']], 
                        [metrics['fn'], metrics['tp']]])
    
    cmap_cm = LinearSegmentedColormap.from_list("custom_cm", [PANEL_COLOR, MALIGNANT_COLOR])
    
    sns.heatmap(cm_data, annot=True, fmt='d', cmap=cmap_cm, ax=ax_cm, cbar=False,
                annot_kws={"size": 32, "weight": "bold"},
                xticklabels=['Predicted Benign', 'Predicted Malignant'],
                yticklabels=['Actual Benign', 'Actual Malignant'])
                
    ax_cm.set_title(f'Triage Performance at T={threshold:.2f} (~100% Sensitivity)', color='white', fontsize=14, fontweight='bold', pad=20)
    
    # Format axes
    ax_cm.tick_params(colors=TEXT_COLOR, labelsize=14)
    
    # Add interpretations
    ax_cm.text(0.5, 0.2, "True Negative\nCorrectly Discharged", ha="center", va="center", color=TEXT_COLOR, fontsize=12)
    ax_cm.text(1.5, 0.2, "False Positive\nOver-Triage (Biopsy)", ha="center", va="center", color="white", fontsize=12)
    if metrics['fn'] == 0:
        ax_cm.text(0.5, 1.2, "False Negative\n(0 Cases - PERFECT)", ha="center", va="center", color=ACCENT_CYAN, fontsize=14, fontweight="bold")
    else:
        ax_cm.text(0.5, 1.2, f"False Negative\n({metrics['fn']} Fatal Omissions)", ha="center", va="center", color="white", fontsize=12, fontweight="bold")
    ax_cm.text(1.5, 1.2, "True Positive\nCorrectly Caught", ha="center", va="center", color=TEXT_COLOR, fontsize=12)
    
    # Save the figure
    dashboard_path = os.path.join(OUTPUT_DIR, 'model_performance_dashboard.png')
    fig.savefig(dashboard_path, dpi=200, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    
    print(f"  ✓ Saved Confusion Matrix: {dashboard_path}")

if __name__ == '__main__':
    generate_dashboard()

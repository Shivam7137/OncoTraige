import os
import json
import numpy as np
import shap
import matplotlib.pyplot as plt
from p1_00_config import ARTIFACT_DIR, INTER_DIR, OUTPUT_DIR, BG_COLOR

def generate_waterfall():
    print("Loading artifacts for SHAP waterfall...")
    X_test = np.load(f'{INTER_DIR}/X_test_final.npy')
    y_test = np.load(f'{INTER_DIR}/y_test.npy')
    prob_test = np.load(f'{INTER_DIR}/rf_y_prob_test.npy')
    
    with open(f'{ARTIFACT_DIR}/all_features.json', 'r') as f:
        features = json.load(f)
        
    shap_values = np.load(f'{ARTIFACT_DIR}/shap_values_test.npy')
    base_val = np.load(f'{ARTIFACT_DIR}/shap_base_value.npy')

    # Select a patient who is Malignant (y_test == 1) and confidently predicted
    idx = np.where((y_test.squeeze() == 1) & (prob_test.squeeze() > 0.8))[0]
    if len(idx) > 0:
        patient_idx = idx[0]
    else:
        patient_idx = 0
        
    print(f"Generating waterfall for patient {patient_idx} (True: {y_test[patient_idx]}, Prob: {prob_test[patient_idx]:.3f})")

    # Extract base value for the Malignant class if it's an array
    if isinstance(base_val, np.ndarray):
        if len(base_val) == 2:
            bv = float(base_val[1])
        elif len(base_val) > 0:
            bv = float(base_val[-1])
        else:
            bv = 0.0
    else:
        bv = float(base_val)

    # Create Explanation object
    exp = shap.Explanation(
        values=shap_values[patient_idx],
        base_values=bv,
        data=X_test[patient_idx],
        feature_names=features
    )
    
    # Generate plot
    shap.plots.waterfall(exp, show=False, max_display=10)
    
    fig = plt.gcf()
    fig.patch.set_facecolor(BG_COLOR)
    
    # Adjust typography colors for the dark theme manually
    axes = fig.axes
    for ax in axes:
        ax.set_facecolor(BG_COLOR)
        ax.tick_params(colors='white')
        ax.xaxis.label.set_color('white')
        ax.yaxis.label.set_color('white')
        ax.title.set_color('white')
        for spine in ax.spines.values():
            spine.set_edgecolor('white')
            
    # Also adjust text annotations
    for text in fig.findobj(match=plt.Text):
        color = text.get_color()
        is_black = False
        if isinstance(color, str):
            if color.lower() == 'black' or color == '#000000':
                is_black = True
        elif hasattr(color, '__len__') and len(color) >= 3:
            if color[0] == 0.0 and color[1] == 0.0 and color[2] == 0.0:
                is_black = True
                
        if is_black:
            text.set_color('white')

    # Save
    plot_path = os.path.join(OUTPUT_DIR, 'shap_waterfall.png')
    plt.savefig(plot_path, bbox_inches='tight', dpi=200, facecolor=BG_COLOR)
    plt.close()
    
    print(f"  ✓ Saved SHAP waterfall plot to {plot_path}")

if __name__ == '__main__':
    generate_waterfall()

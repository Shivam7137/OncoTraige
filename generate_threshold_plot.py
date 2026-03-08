import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from p1_00_config import (
    INTER_DIR, OUTPUT_DIR, THRESHOLD, BENIGN_COLOR, MALIGNANT_COLOR,
    BG_COLOR, PANEL_COLOR, TEXT_COLOR, GRID_COLOR, ACCENT_CYAN,
    save_figure, section_header
)

def generate_threshold_plot():
    print("Generating Threshold Performance Plot...")
    
    # Load test probabilities and true labels
    y_test = np.load(os.path.join(INTER_DIR, 'y_test.npy'))
    prob_test = np.load(os.path.join(INTER_DIR, 'rf_y_prob_test.npy'))
    
    thresholds = np.linspace(0.01, 0.99, 100)
    accuracies = []
    precisions = []
    recalls = []
    f1_scores = []
    
    for t in thresholds:
        y_pred = (prob_test >= t).astype(int)
        accuracies.append(accuracy_score(y_test, y_pred))
        # Zero_division=0 to handle extreme thresholds where no positive predictions exist
        precisions.append(precision_score(y_test, y_pred, zero_division=0))
        recalls.append(recall_score(y_test, y_pred))
        f1_scores.append(f1_score(y_test, y_pred, zero_division=0))
        
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.plot(thresholds, accuracies, label='Accuracy', color=BENIGN_COLOR, linewidth=2)
    ax.plot(thresholds, precisions, label='Precision', color=ACCENT_CYAN, linewidth=2)
    ax.plot(thresholds, recalls, label='Recall', color=MALIGNANT_COLOR, linewidth=2)
    ax.plot(thresholds, f1_scores, label='F1 Score', color='#FF9800', linewidth=2)
    
    # Add vertical line for the selected threshold
    ax.axvline(x=THRESHOLD, color='white', linestyle='--', linewidth=2, label=f'Selected Threshold ({THRESHOLD:.2f})')
    
    ax.set_title('Model Performance Across Decision Thresholds\nOptimal Balance between Sensitivity and Specificity', pad=20, fontweight='bold', fontsize=14)
    ax.set_xlabel('Decision Threshold (Probability of Malignancy)', labelpad=10, fontsize=12)
    ax.set_ylabel('Score', labelpad=10, fontsize=12)
    
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.05)
    
    # Style legend
    ax.legend(loc='lower left', framealpha=0.8, facecolor=PANEL_COLOR, edgecolor=GRID_COLOR, fontsize=10)
    
    # Cleanup spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # Save the figure
    # Using save_figure from config which puts it in OUTPUT_DIR
    save_figure(fig, 'threshold_performance.png')
    
if __name__ == '__main__':
    generate_threshold_plot()

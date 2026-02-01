import json
from typing import Dict, List

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    auc,
    confusion_matrix,
    precision_recall_curve,
    roc_curve
)

# Enforce consistent reporting visuals for stakeholder review
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10


def plot_roc_curve(y_true: np.ndarray, y_pred_proba: np.ndarray, save_path: str = None):
    """
    Plot ROC curve with AUC score.
    
    Args:
        y_true: True labels
        y_pred_proba: Predicted probabilities
        save_path: Path to save figure
    """
    fpr, tpr, thresholds = roc_curve(y_true, y_pred_proba)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(10, 8))
    plt.plot(fpr, tpr, color='darkorange', lw=2, 
             label=f'ROC Curve (AUC = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', 
             label='Random Classifier')
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('Receiver Operating Characteristic (ROC) Curve', fontsize=14, fontweight='bold')
    plt.legend(loc="lower right", fontsize=11)
    plt.grid(alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved: {save_path}")
    
    plt.close()


def plot_precision_recall_curve(y_true: np.ndarray, y_pred_proba: np.ndarray, save_path: str = None):
    """
    Plot Precision-Recall curve with AUC score.
    
    Args:
        y_true: True labels
        y_pred_proba: Predicted probabilities
        save_path: Path to save figure
    """
    precision, recall, thresholds = precision_recall_curve(y_true, y_pred_proba)
    pr_auc = auc(recall, precision)
    
    plt.figure(figsize=(10, 8))
    plt.plot(recall, precision, color='blue', lw=2, 
             label=f'PR Curve (AUC = {pr_auc:.4f})')
    
    # Anchor comparison to naive precision baseline
    baseline = y_true.sum() / len(y_true)
    plt.axhline(y=baseline, color='navy', linestyle='--', lw=2,
                label=f'Random Classifier (Precision = {baseline:.4f})')
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall', fontsize=12)
    plt.ylabel('Precision', fontsize=12)
    plt.title('Precision-Recall Curve', fontsize=14, fontweight='bold')
    plt.legend(loc="upper right", fontsize=11)
    plt.grid(alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved: {save_path}")
    
    plt.close()


def plot_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, save_path: str = None):
    """
    Plot confusion matrix heatmap.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        save_path: Path to save figure
    """
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Legitimate', 'Fraud'],
                yticklabels=['Legitimate', 'Fraud'],
                cbar_kws={'label': 'Count'})
    
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.title('Confusion Matrix', fontsize=14, fontweight='bold')
    
    # Enforce percentage readability for stakeholders
    total = cm.sum()
    for i in range(2):
        for j in range(2):
            percentage = (cm[i, j] / total) * 100
            plt.text(j + 0.5, i + 0.7, f'({percentage:.2f}%)', 
                    ha='center', va='center', fontsize=9, color='gray')
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved: {save_path}")
    
    plt.close()


def plot_feature_importance(model, feature_names: List[str], top_n: int = 20, save_path: str = None):
    """
    Plot feature importance bar chart.
    
    Args:
        model: Trained XGBoost model
        feature_names: List of feature names
        top_n: Number of top features to display
        save_path: Path to save figure
    """
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False).head(top_n)
    
    plt.figure(figsize=(10, 8))
    sns.barplot(data=importance_df, y='feature', x='importance', palette='viridis')
    
    plt.xlabel('Importance Score', fontsize=12)
    plt.ylabel('Feature', fontsize=12)
    plt.title(f'Top {top_n} Feature Importance', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved: {save_path}")
    
    plt.close()


def plot_threshold_analysis(y_true: np.ndarray, y_pred_proba: np.ndarray, save_path: str = None):
    """
    Plot precision, recall, and F1-score vs threshold.
    
    Args:
        y_true: True labels
        y_pred_proba: Predicted probabilities
        save_path: Path to save figure
    """
    precision, recall, thresholds = precision_recall_curve(y_true, y_pred_proba)
    
    # Enforce balanced scorecard for threshold selection
    f1_scores = 2 * (precision * recall) / (precision + recall + 1e-10)
    
    # Select threshold that balances precision/recall
    optimal_idx = np.argmax(f1_scores)
    optimal_threshold = thresholds[optimal_idx] if optimal_idx < len(thresholds) else 0.5
    
    plt.figure(figsize=(12, 6))
    
    # Align arrays to ensure stable plotting
    precision = precision[:-1]
    recall = recall[:-1]
    f1_scores = f1_scores[:-1]
    
    plt.plot(thresholds, precision, label='Precision', lw=2)
    plt.plot(thresholds, recall, label='Recall', lw=2)
    plt.plot(thresholds, f1_scores, label='F1 Score', lw=2)
    
    plt.axvline(x=optimal_threshold, color='red', linestyle='--', lw=2,
                label=f'Optimal Threshold = {optimal_threshold:.3f}')
    
    plt.xlabel('Threshold', fontsize=12)
    plt.ylabel('Score', fontsize=12)
    plt.title('Precision, Recall, and F1-Score vs Threshold', fontsize=14, fontweight='bold')
    plt.legend(loc='best', fontsize=11)
    plt.grid(alpha=0.3)
    plt.xlim([0, 1])
    plt.ylim([0, 1.05])
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved: {save_path}")
    
    plt.close()
    
    return optimal_threshold


def generate_evaluation_report(model_path: str, data_path: str, output_dir: str = "./reports"):
    """
    Generate comprehensive evaluation report with visualizations.
    
    Args:
        model_path: Path to trained model
        data_path: Path to evaluation data
        output_dir: Directory to save reports
    """
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    print("\n" + "=" * 80)
    print("SENTINEL MODEL EVALUATION REPORT")
    print("=" * 80)
    
    # Enforce artifact loading for evaluation
    print("\n[1/6] Loading model...")
    model = joblib.load(model_path)
    print(f"✓ Loaded: {model_path}")
    
    # Enforce evaluation data loading
    print("\n[2/6] Loading evaluation data...")
    df = pd.read_pickle(data_path)
    print(f"✓ Loaded: {data_path}")
    print(f"  Shape: {df.shape}")
    
    # Enforce feature alignment with training schema
    from model_training import prepare_features_and_target
    X, y = prepare_features_and_target(df)
    
    # Enforce feature list consistency for reporting
    feature_names_path = model_path.replace('.pkl', '').replace('model', 'feature_names') + '.json'
    try:
        with open('./models/feature_names.json') as f:
            feature_names = json.load(f)
    except:
        feature_names = X.columns.tolist()
    
    # Enforce probability scoring for evaluation
    print("\n[3/6] Generating predictions...")
    y_pred_proba = model.predict_proba(X)[:, 1]
    y_pred = (y_pred_proba >= 0.5).astype(int)
    
    # Emit plots for stakeholder review
    print("\n[4/6] Generating visualizations...")
    
    plot_roc_curve(y, y_pred_proba, f"{output_dir}/roc_curve.png")
    plot_precision_recall_curve(y, y_pred_proba, f"{output_dir}/precision_recall_curve.png")
    plot_confusion_matrix(y, y_pred, f"{output_dir}/confusion_matrix.png")
    plot_feature_importance(model, feature_names, top_n=20, save_path=f"{output_dir}/feature_importance.png")
    optimal_threshold = plot_threshold_analysis(y, y_pred_proba, f"{output_dir}/threshold_analysis.png")
    
    # Emit metrics for governance review
    print("\n[5/6] Calculating metrics...")
    
    from sklearn.metrics import classification_report, roc_auc_score
    
    metrics = {
        'roc_auc': roc_auc_score(y, y_pred_proba),
        'optimal_threshold': float(optimal_threshold),
        'classification_report': classification_report(y, y_pred, target_names=['Legitimate', 'Fraud'])
    }
    
    # Persist evaluation artifacts
    print("\n[6/6] Saving evaluation report...")
    
    report_path = f"{output_dir}/evaluation_report.txt"
    with open(report_path, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("SENTINEL FRAUD DETECTION - MODEL EVALUATION REPORT\n")
        f.write("=" * 80 + "\n\n")
        
        f.write(f"Model: {model_path}\n")
        f.write(f"Data: {data_path}\n")
        f.write(f"Evaluation Date: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("-" * 80 + "\n")
        f.write("PERFORMANCE METRICS\n")
        f.write("-" * 80 + "\n\n")
        
        f.write(f"ROC-AUC Score: {metrics['roc_auc']:.4f}\n")
        f.write(f"Optimal Threshold: {metrics['optimal_threshold']:.3f}\n\n")
        
        f.write("Classification Report:\n")
        f.write(metrics['classification_report'])
        f.write("\n")
        
        f.write("-" * 80 + "\n")
        f.write("GENERATED VISUALIZATIONS\n")
        f.write("-" * 80 + "\n\n")
        
        f.write(f"1. ROC Curve: {output_dir}/roc_curve.png\n")
        f.write(f"2. Precision-Recall Curve: {output_dir}/precision_recall_curve.png\n")
        f.write(f"3. Confusion Matrix: {output_dir}/confusion_matrix.png\n")
        f.write(f"4. Feature Importance: {output_dir}/feature_importance.png\n")
        f.write(f"5. Threshold Analysis: {output_dir}/threshold_analysis.png\n")
    
    print(f"✓ Saved: {report_path}")
    
    print("\n" + "=" * 80)
    print("EVALUATION COMPLETE")
    print("=" * 80)
    print(f"\n📊 ROC-AUC: {metrics['roc_auc']:.4f}")
    print(f"🎯 Optimal Threshold: {metrics['optimal_threshold']:.3f}")
    print(f"\n📁 Reports saved to: {output_dir}/")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    # Enforce plotting dependency presence
    try:
        import matplotlib
        import seaborn
    except ImportError:
        print("\n⚠ ERROR: Visualization libraries not installed")
        print("\nInstall with:")
        print("  pip install matplotlib seaborn")
        exit(1)
    
    # Execute end-to-end evaluation workflow
    generate_evaluation_report(
        model_path="./models/sentinel_fraud_model.pkl",
        data_path="./data/train_engineered.pkl",
        output_dir="./reports"
    )

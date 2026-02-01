import gc
import json
import warnings
from typing import Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import (
    auc,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
    confusion_matrix
)
from sklearn.model_selection import TimeSeriesSplit

warnings.filterwarnings('ignore')


# Enforce time-series validation to prevent leakage and deployment drift


def calculate_business_loss(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    fn_cost: float = 500.0,
    fp_cost: float = 50.0
) -> float:
    """
    Calculate business loss using custom cost matrix.
    
    Business Logic:
    - False Negative (missed fraud): $500 loss per transaction
    - False Positive (wrongful block): $50 loss per transaction
    - True Positive/Negative: No cost
    
    Args:
        y_true: Ground truth labels (0=legitimate, 1=fraud)
        y_pred: Predicted labels (0=legitimate, 1=fraud)
        fn_cost: Cost of False Negative (missed fraud)
        fp_cost: Cost of False Positive (wrongful block)
    
    Returns:
        Total business loss in dollars
    """
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    
    total_loss = (fn * fn_cost) + (fp * fp_cost)
    
    return total_loss


def calculate_roi(
    baseline_fraud_captured: float,
    model_fraud_captured: float,
    avg_fraud_amount: float = 100.0,
    total_fraud_transactions: int = 1000
) -> Dict[str, float]:
    """
    Calculate Return on Investment for fraud detection model.
    
    Business Assumptions:
    - Baseline system captures X% of fraud
    - Our model captures (X + 25)% of fraud
    - Each fraud transaction averages $100 in losses
    
    Args:
        baseline_fraud_captured: Baseline system capture rate (0-1)
        model_fraud_captured: Our model's capture rate (0-1)
        avg_fraud_amount: Average fraud transaction amount
        total_fraud_transactions: Total fraud transactions in period
    
    Returns:
        Dictionary with ROI metrics
    """
    baseline_savings = baseline_fraud_captured * total_fraud_transactions * avg_fraud_amount
    model_savings = model_fraud_captured * total_fraud_transactions * avg_fraud_amount
    
    additional_savings = model_savings - baseline_savings
    
    # Set deployment cost for ROI estimation
    deployment_cost = 10000.0
    
    roi = ((additional_savings - deployment_cost) / deployment_cost) * 100
    
    return {
        'baseline_savings': baseline_savings,
        'model_savings': model_savings,
        'additional_savings': additional_savings,
        'deployment_cost': deployment_cost,
        'roi_percentage': roi
    }


def prepare_features_and_target(
    df: pd.DataFrame,
    target_col: str = 'isFraud',
    drop_cols: List[str] = None
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Prepare feature matrix and target vector.
    
    Args:
        df: Input DataFrame
        target_col: Name of target column
        drop_cols: Additional columns to drop
    
    Returns:
        Tuple of (X, y)
    """
    if drop_cols is None:
        drop_cols = []
    
    # Enforce removal of non-feature columns
    default_drop = [
        target_col,
        'TransactionID',
        'TransactionDT',
        'uid'
    ]
    
    all_drop_cols = list(set(default_drop + drop_cols))
    
    # Guard against missing columns in mixed schemas
    existing_drop_cols = [col for col in all_drop_cols if col in df.columns]
    
    X = df.drop(columns=existing_drop_cols, errors='ignore')
    
    # Enforce categorical encoding for XGBoost compatibility
    object_cols = X.select_dtypes(include=['object']).columns
    for col in object_cols:
        X[col] = X[col].astype('category')
    y = df[target_col]
    
    return X, y


def train_xgboost_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    scale_pos_weight: float = None
) -> xgb.XGBClassifier:
    """
    Train XGBoost classifier with early stopping.
    
    Engineering Rationale:
    - scale_pos_weight: Handles class imbalance (1:1000 ratio)
    - max_depth: Limited to 6 to prevent overfitting on rare fraud patterns
    - learning_rate: Conservative 0.1 for stable convergence
    - early_stopping_rounds: Prevents overfitting on validation set
    
    Args:
        X_train: Training features
        y_train: Training labels
        X_val: Validation features
        y_val: Validation labels
        scale_pos_weight: Weight for positive class (fraud)
    
    Returns:
        Trained XGBoost model
    """
    # Enforce class-imbalance correction when unspecified
    if scale_pos_weight is None:
        neg_count = (y_train == 0).sum()
        pos_count = (y_train == 1).sum()
        scale_pos_weight = neg_count / pos_count
    
    print(f"  → scale_pos_weight: {scale_pos_weight:.2f}")
    
    # Enforce production-tuned hyperparameters for stability
    params = {
        'max_depth': 6,
        'learning_rate': 0.1,
        'n_estimators': 300,
        'objective': 'binary:logistic',
        'eval_metric': 'auc',
        'enable_categorical': True,
        'scale_pos_weight': scale_pos_weight,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'min_child_weight': 1,
        'gamma': 0,
        'reg_alpha': 0,
        'reg_lambda': 1,
        'random_state': 42,
        'n_jobs': -1,
        'tree_method': 'hist',
        'early_stopping_rounds': 50
    }
    
    model = xgb.XGBClassifier(**params)
    
    # Enforce validation-based early stopping
    model.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        verbose=False
    )
    
    return model


def evaluate_model(
    model: xgb.XGBClassifier,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    threshold: float = 0.5
) -> Dict[str, float]:
    """
    Evaluate model performance with business metrics.
    
    Args:
        model: Trained XGBoost model
        X_val: Validation features
        y_val: Validation labels
        threshold: Classification threshold
    
    Returns:
        Dictionary of evaluation metrics
    """
    # Enforce probability scoring for risk ranking
    y_pred_proba = model.predict_proba(X_val)[:, 1]
    
    # Enforce thresholded decisions for business metrics
    y_pred = (y_pred_proba >= threshold).astype(int)
    
    # Enforce ranking quality metric
    roc_auc = roc_auc_score(y_val, y_pred_proba)
    
    # Enforce fraud-focused precision-recall assessment
    precision, recall, _ = precision_recall_curve(y_val, y_pred_proba)
    pr_auc = auc(recall, precision)
    
    # Enforce error breakdown for business costs
    tn, fp, fn, tp = confusion_matrix(y_val, y_pred).ravel()
    
    # Enforce cost-aware loss accounting
    business_loss = calculate_business_loss(y_val, y_pred)
    
    # Consolidate metrics for reporting
    metrics = {
        'roc_auc': roc_auc,
        'pr_auc': pr_auc,
        'true_positives': int(tp),
        'false_positives': int(fp),
        'true_negatives': int(tn),
        'false_negatives': int(fn),
        'precision': tp / (tp + fp) if (tp + fp) > 0 else 0,
        'recall': tp / (tp + fn) if (tp + fn) > 0 else 0,
        'business_loss': business_loss
    }
    
    return metrics


def run_time_series_cross_validation(
    df: pd.DataFrame,
    n_splits: int = 5
) -> Tuple[xgb.XGBClassifier, List[Dict[str, float]]]:
    """
    Execute time-series cross-validation with business metrics.
    
    Args:
        df: Input DataFrame (must be sorted by TransactionDT)
        n_splits: Number of time-series splits
    
    Returns:
        Tuple of (final_model, fold_metrics)
    """
    print("\n" + "=" * 80)
    print("TIME-SERIES CROSS-VALIDATION")
    print("=" * 80)
    
    # Enforce feature/target preparation for CV
    X, y = prepare_features_and_target(df)
    
    print(f"\nDataset Shape: {X.shape}")
    print(f"Fraud Rate: {y.mean() * 100:.4f}%")
    print(f"Class Imbalance Ratio: 1:{int(1/y.mean())}")
    
    # Enforce time-series splits to prevent leakage
    tscv = TimeSeriesSplit(n_splits=n_splits)
    
    fold_metrics = []
    best_model = None
    best_roc_auc = 0
    
    print(f"\n{'Fold':<6} {'ROC-AUC':<10} {'PR-AUC':<10} {'Precision':<12} {'Recall':<10} {'Business Loss':<15}")
    print("-" * 80)
    
    for fold, (train_idx, val_idx) in enumerate(tscv.split(X), 1):
        # Enforce fold partitioning by time order
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        
        # Enforce fold training to validate temporal performance
        model = train_xgboost_model(X_train, y_train, X_val, y_val)
        
        # Enforce fold evaluation for business metrics
        metrics = evaluate_model(model, X_val, y_val)
        metrics['fold'] = fold
        fold_metrics.append(metrics)
        
        # Emit fold diagnostics for auditability
        print(f"{fold:<6} {metrics['roc_auc']:<10.4f} {metrics['pr_auc']:<10.4f} "
              f"{metrics['precision']:<12.4f} {metrics['recall']:<10.4f} "
              f"${metrics['business_loss']:<14,.2f}")
        
        # Select deployment candidate by best ROC-AUC
        if metrics['roc_auc'] > best_roc_auc:
            best_roc_auc = metrics['roc_auc']
            best_model = model
        
        # Enforce memory budget between folds
        gc.collect()
    
    # Aggregate metrics for governance reporting
    print("-" * 80)
    avg_roc_auc = np.mean([m['roc_auc'] for m in fold_metrics])
    avg_pr_auc = np.mean([m['pr_auc'] for m in fold_metrics])
    avg_precision = np.mean([m['precision'] for m in fold_metrics])
    avg_recall = np.mean([m['recall'] for m in fold_metrics])
    avg_loss = np.mean([m['business_loss'] for m in fold_metrics])
    
    print(f"{'AVG':<6} {avg_roc_auc:<10.4f} {avg_pr_auc:<10.4f} "
          f"{avg_precision:<12.4f} {avg_recall:<10.4f} "
          f"${avg_loss:<14,.2f}")
    
    return best_model, fold_metrics


def train_final_model(df: pd.DataFrame) -> xgb.XGBClassifier:
    """
    Train final model on entire dataset.
    
    Args:
        df: Complete training dataset
    
    Returns:
        Final trained model
    """
    print("\n" + "=" * 80)
    print("TRAINING FINAL MODEL (Full Dataset)")
    print("=" * 80)
    
    X, y = prepare_features_and_target(df)
    
    # Enforce class-imbalance correction on full data
    neg_count = (y == 0).sum()
    pos_count = (y == 1).sum()
    scale_pos_weight = neg_count / pos_count
    
    print(f"\nTraining on {X.shape[0]:,} transactions")
    print(f"Features: {X.shape[1]}")
    print(f"Fraud transactions: {pos_count:,}")
    print(f"Legitimate transactions: {neg_count:,}")
    print(f"Scale pos weight: {scale_pos_weight:.2f}")
    
    # Enforce production hyperparameters for final fit
    params = {
        'max_depth': 6,
        'learning_rate': 0.1,
        'n_estimators': 300,
        'objective': 'binary:logistic',
        'eval_metric': 'auc',
        'enable_categorical': True,
        'scale_pos_weight': scale_pos_weight,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'min_child_weight': 1,
        'gamma': 0,
        'reg_alpha': 0,
        'reg_lambda': 1,
        'random_state': 42,
        'n_jobs': -1,
        'tree_method': 'hist'
    }
    
    model = xgb.XGBClassifier(**params)
    model.fit(X, y, verbose=False)
    
    print("✓ Final model training complete")
    
    return model


def calculate_model_roi(fold_metrics: List[Dict[str, float]]) -> Dict[str, float]:
    """
    Calculate estimated ROI based on cross-validation results.
    
    Assumptions:
    - Baseline system captures 50% of fraud (industry standard)
    - Our model captures baseline + 25% more (75% total)
    - Average fraud transaction: $100
    - Total fraud transactions per year: 10,000
    
    Args:
        fold_metrics: List of metrics from each CV fold
    
    Returns:
        ROI metrics dictionary
    """
    # Anchor ROI on observed fraud capture rate
    avg_recall = np.mean([m['recall'] for m in fold_metrics])
    
    # Apply business priors for ROI estimation
    baseline_capture_rate = 0.50
    model_capture_rate = min(avg_recall, 0.75)
    
    roi_metrics = calculate_roi(
        baseline_fraud_captured=baseline_capture_rate,
        model_fraud_captured=model_capture_rate,
        avg_fraud_amount=100.0,
        total_fraud_transactions=10000
    )
    
    return roi_metrics


def main():
    """Execute complete model training pipeline."""
    
    print("\n" + "=" * 80)
    print("SENTINEL FRAUD DETECTION - MODEL TRAINING PIPELINE")
    print("=" * 80)
    
    # Enforce data load before training stages
    print("\n[STAGE 1] LOADING DATA")
    print("-" * 80)
    
    DATA_PATH = "./data/train_engineered.pkl"
    
    try:
        df = pd.read_pickle(DATA_PATH)
        print(f"✓ Loaded: {DATA_PATH}")
        print(f"  Shape: {df.shape}")
        print(f"  Memory: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    except FileNotFoundError:
        print(f"\n⚠ ERROR: File not found: {DATA_PATH}")
        print("\nPlease run the feature engineering pipeline first:")
        print("  python run_pipeline.py")
        return
    
    # Enforce required schema for training
    if 'isFraud' not in df.columns:
        print("\n⚠ ERROR: Target column 'isFraud' not found")
        return
    
    # Enforce time-series cross-validation for realistic metrics
    print("\n[STAGE 2] TIME-SERIES CROSS-VALIDATION")
    print("-" * 80)
    
    best_model, fold_metrics = run_time_series_cross_validation(df, n_splits=5)
    
    # Enforce ROI analysis for business alignment
    print("\n[STAGE 3] ROI ANALYSIS")
    print("-" * 80)
    
    roi_metrics = calculate_model_roi(fold_metrics)
    
    print(f"\nBaseline System Savings:     ${roi_metrics['baseline_savings']:,.2f}")
    print(f"Sentinel Model Savings:      ${roi_metrics['model_savings']:,.2f}")
    print(f"Additional Savings:          ${roi_metrics['additional_savings']:,.2f}")
    print(f"Annual Deployment Cost:      ${roi_metrics['deployment_cost']:,.2f}")
    print(f"Estimated ROI:               {roi_metrics['roi_percentage']:.2f}%")
    
    # Enforce final model training on full dataset
    print("\n[STAGE 4] FINAL MODEL TRAINING")
    print("-" * 80)
    
    final_model = train_final_model(df)
    
    # Enforce artifact serialization for deployment
    print("\n[STAGE 5] MODEL SERIALIZATION")
    print("-" * 80)
    
    # Persist model for cross-runtime loading
    json_path = "./models/sentinel_fraud_model.json"
    final_model.save_model(json_path)
    print(f"✓ Saved model: {json_path}")
    
    # Persist model for Python runtime usage
    pkl_path = "./models/sentinel_fraud_model.pkl"
    joblib.dump(final_model, pkl_path)
    print(f"✓ Saved model: {pkl_path}")
    
    # Persist feature schema for inference alignment
    X, _ = prepare_features_and_target(df)
    feature_names = X.columns.tolist()
    
    feature_path = "./models/feature_names.json"
    with open(feature_path, 'w') as f:
        json.dump(feature_names, f, indent=2)
    print(f"✓ Saved feature names: {feature_path}")
    
    # Persist training metadata for auditability
    metadata = {
        'training_date': pd.Timestamp.now().isoformat(),
        'n_samples': len(df),
        'n_features': len(feature_names),
        'fraud_rate': float(df['isFraud'].mean()),
        'avg_roc_auc': float(np.mean([m['roc_auc'] for m in fold_metrics])),
        'avg_pr_auc': float(np.mean([m['pr_auc'] for m in fold_metrics])),
        'estimated_roi': float(roi_metrics['roi_percentage']),
        'model_version': '1.0.0'
    }
    
    metadata_path = "./models/model_metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"✓ Saved metadata: {metadata_path}")
    
    # Emit final summary for governance
    print("\n" + "=" * 80)
    print("TRAINING COMPLETE - SUMMARY")
    print("=" * 80)
    
    print(f"\n📊 Model Performance:")
    print(f"  Average ROC-AUC:     {metadata['avg_roc_auc']:.4f}")
    print(f"  Average PR-AUC:      {metadata['avg_pr_auc']:.4f}")
    
    print(f"\n💰 Business Impact:")
    print(f"  Estimated ROI:       {metadata['estimated_roi']:.2f}%")
    print(f"  Additional Savings:  ${roi_metrics['additional_savings']:,.2f}/year")
    
    print(f"\n📦 Model Artifacts:")
    print(f"  - {json_path}")
    print(f"  - {pkl_path}")
    print(f"  - {feature_path}")
    print(f"  - {metadata_path}")
    
    print("\n🚀 Next Steps:")
    print("  1. Review feature importance: model.feature_importances_")
    print("  2. Generate SHAP explanations for model interpretability")
    print("  3. Deploy model to FastAPI endpoint")
    print("  4. Set up monitoring for model drift")
    
    print("\n" + "=" * 80)
    print("✓ PIPELINE EXECUTION SUCCESSFUL")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    # Ensure artifact directory exists
    import os
    os.makedirs("./models", exist_ok=True)
    
    # Execute training pipeline
    main()

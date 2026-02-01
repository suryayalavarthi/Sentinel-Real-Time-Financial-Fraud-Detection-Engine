"""
Quick Pipeline Execution for Phase 1 Operationalization
Runs data ingestion, feature engineering, and model training with reduced dataset
"""

import pandas as pd
import numpy as np
from data_ingestion import load_and_optimize_data
from feature_engineering import run_feature_engineering_pipeline
from model_training import run_time_series_cross_validation, train_final_model, prepare_features_and_target
import os

print("\n" + "=" * 80)
print("PHASE 1: QUICK PIPELINE EXECUTION")
print("=" * 80)

# Create directories
os.makedirs("./models", exist_ok=True)
os.makedirs("./reports", exist_ok=True)
os.makedirs("./logs", exist_ok=True)

# Step 1: Data Ingestion (sample for speed)
print("\n[1/3] Data Ingestion (Sampling for speed)...")
try:
    # Load full data
    train_df, test_df = load_and_optimize_data(
        train_transaction_path="./data/train_transaction.csv",
        train_identity_path="./data/train_identity.csv",
        test_transaction_path="./data/test_transaction.csv",
        test_identity_path="./data/test_identity.csv"
    )
    
    # Sample for quick execution (50K rows)
    print(f"  Original size: {len(train_df):,} rows")
    train_df = train_df.sample(n=min(50000, len(train_df)), random_state=42)
    print(f"  Sampled to: {len(train_df):,} rows for quick execution")
    
    # Save
    train_df.to_pickle("./data/train_optimized.pkl")
    test_df.to_pickle("./data/test_optimized.pkl")
    print("  ✓ Data ingestion complete")
    
except Exception as e:
    print(f"  ✗ Error: {e}")
    raise

# Step 2: Feature Engineering
print("\n[2/3] Feature Engineering...")
try:
    train_df = run_feature_engineering_pipeline(train_df)
    train_df.to_pickle("./data/train_engineered.pkl")
    print("  ✓ Feature engineering complete")
except Exception as e:
    print(f"  ✗ Error: {e}")
    raise

# Step 3: Model Training
print("\n[3/3] Model Training (Quick)...")
try:
    # Train with 3 folds for speed
    best_model, fold_metrics = run_time_series_cross_validation(train_df, n_splits=3)
    
    # Train final model
    final_model = train_final_model(train_df)
    
    print("  ✓ Model training complete")
    print(f"  Average ROC-AUC: {np.mean([m['roc_auc'] for m in fold_metrics]):.4f}")
    
except Exception as e:
    print(f"  ✗ Error: {e}")
    raise

print("\n" + "=" * 80)
print("PIPELINE EXECUTION COMPLETE")
print("=" * 80)
print("\n✓ Ready for interpretability analysis")
print(f"  - Model: ./models/sentinel_fraud_model.pkl")
print(f"  - Data: ./data/train_engineered.pkl")
print("\n" + "=" * 80 + "\n")

import pandas as pd
import gc
from data_ingestion import load_and_optimize_data
from feature_engineering import run_feature_engineering_pipeline


def main():
    """Execute complete Sentinel pipeline."""
    
    print("\n" + "=" * 80)
    print("SENTINEL FRAUD DETECTION - COMPLETE PIPELINE")
    print("=" * 80)
    
    # Enforce ingestion-first ordering for reproducibility
    print("\n[STAGE 1] DATA INGESTION")
    print("-" * 80)
    
    RAW_DATA_DIR = "./data/ieee-fraud-detection/"
    PROCESSED_DATA_DIR = "./data/"
    
    try:
        train_df, test_df = load_and_optimize_data(
            train_transaction_path=f"{RAW_DATA_DIR}train_transaction.csv",
            train_identity_path=f"{RAW_DATA_DIR}train_identity.csv",
            test_transaction_path=f"{RAW_DATA_DIR}test_transaction.csv",
            test_identity_path=f"{RAW_DATA_DIR}test_identity.csv"
        )
    except FileNotFoundError as e:
        print(f"\n⚠ ERROR: {e}")
        print("\nPlease ensure the following files exist in ./data/ieee-fraud-detection/:")
        print("  - train_transaction.csv")
        print("  - train_identity.csv")
        print("  - test_transaction.csv")
        print("  - test_identity.csv")
        print("\nDownload from: https://www.kaggle.com/c/ieee-fraud-detection/data")
        return
    
    # Enforce feature engineering after ingestion
    print("\n[STAGE 2] FEATURE ENGINEERING - TRAINING SET")
    print("-" * 80)
    
    train_df = run_feature_engineering_pipeline(train_df)
    
    print("\n[STAGE 2] FEATURE ENGINEERING - TEST SET")
    print("-" * 80)
    
    test_df = run_feature_engineering_pipeline(test_df)
    
    # Persist engineered data for downstream training and monitoring
    print("\n[STAGE 3] SAVING PROCESSED DATA")
    print("-" * 80)
    
    print("Saving training set...")
    train_df.to_pickle(f"{PROCESSED_DATA_DIR}train_engineered.pkl")
    print(f"  ✓ Saved: {PROCESSED_DATA_DIR}train_engineered.pkl")
    
    print("Saving test set...")
    test_df.to_pickle(f"{PROCESSED_DATA_DIR}test_engineered.pkl")
    print(f"  ✓ Saved: {PROCESSED_DATA_DIR}test_engineered.pkl")
    
    # Emit lightweight sample for manual review
    print("\nOptional: Saving sample as CSV for inspection...")
    train_df.head(1000).to_csv(f"{PROCESSED_DATA_DIR}train_sample.csv", index=False)
    print(f"  ✓ Saved: {PROCESSED_DATA_DIR}train_sample.csv (first 1000 rows)")
    
    # Emit pipeline summary for auditability
    print("\n" + "=" * 80)
    print("PIPELINE COMPLETE - SUMMARY REPORT")
    print("=" * 80)
    
    print("\n📊 Training Set:")
    print(f"  Shape: {train_df.shape}")
    print(f"  Memory: {train_df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    print(f"  Features: {train_df.shape[1]}")
    
    print("\n📊 Test Set:")
    print(f"  Shape: {test_df.shape}")
    print(f"  Memory: {test_df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    print(f"  Features: {test_df.shape[1]}")
    
    print("\n🎯 Engineered Features:")
    engineered_features = [
        'uid',
        'uid_TransactionFreq_24h',
        'uid_TransactionAmt_mean_30d',
        'Amt_to_Mean_Ratio',
        'card1_freq',
        'P_emaildomain_freq'
    ]
    
    for feat in engineered_features:
        if feat in train_df.columns:
            print(f"  ✓ {feat}")
    
    print("\n💾 Output Files:")
    print(f"  - {PROCESSED_DATA_DIR}train_engineered.pkl")
    print(f"  - {PROCESSED_DATA_DIR}test_engineered.pkl")
    print(f"  - {PROCESSED_DATA_DIR}train_sample.csv")
    
    print("\n🚀 Next Steps:")
    print("  1. Load processed data: pd.read_pickle(\1../data/processed/train_engineered.pkl')")
    print("  2. Perform EDA on engineered features")
    print("  3. Train fraud detection model (XGBoost, LightGBM, etc.)")
    print("  4. Evaluate with time-series cross-validation")
    
    print("\n" + "=" * 80)
    print("✓ ALL STAGES COMPLETED SUCCESSFULLY")
    print("=" * 80 + "\n")
    
    # Enforce memory hygiene before exit
    gc.collect()


if __name__ == "__main__":
    main()

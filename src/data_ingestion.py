import gc
import pandas as pd
import numpy as np
from typing import Tuple
import warnings

warnings.filterwarnings('ignore')


def reduce_mem_usage(df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    """
    Reduce memory footprint via dtype downcasting.

    Args:
        df: Input DataFrame.
        verbose: Whether to print memory reduction stats.

    Returns:
        Memory-optimized DataFrame.
    """
    start_mem = df.memory_usage(deep=True).sum() / 1024**2
    n_rows = len(df)
    
    for col in df.columns:
        series = df[col]
        col_type = series.dtype
        
        # Enforce categorical compression for low-cardinality text
        if col_type == object:
            unique_count = series.nunique(dropna=False)
            if n_rows > 0 and unique_count / n_rows < 0.5:
                df[col] = series.astype('category', copy=False)
            continue
        
        if pd.api.types.is_bool_dtype(col_type):
            df[col] = series.astype(np.bool_, copy=False)
            continue
        
        null_ratio = series.isna().mean() if n_rows > 0 else 0
        c_min = series.min(skipna=True)
        c_max = series.max(skipna=True)
        has_nulls = series.isna().any()
        
        # Enforce compact integer storage for memory budget
        if pd.api.types.is_integer_dtype(col_type):
            if has_nulls:
                # Preserve missingness while minimizing memory
                if c_min >= 0:
                    if c_max < np.iinfo(np.uint8).max:
                        df[col] = series.astype('UInt8')
                    elif c_max < np.iinfo(np.uint16).max:
                        df[col] = series.astype('UInt16')
                    elif c_max < np.iinfo(np.uint32).max:
                        df[col] = series.astype('UInt32')
                    else:
                        df[col] = series.astype('UInt64')
                else:
                    if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                        df[col] = series.astype('Int8')
                    elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                        df[col] = series.astype('Int16')
                    elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                        df[col] = series.astype('Int32')
                    else:
                        df[col] = series.astype('Int64')
            else:
                # Prefer unsigned ranges when valid to reduce footprint
                if c_min >= 0:
                    if c_max < np.iinfo(np.uint8).max:
                        df[col] = series.astype(np.uint8, copy=False)
                    elif c_max < np.iinfo(np.uint16).max:
                        df[col] = series.astype(np.uint16, copy=False)
                    elif c_max < np.iinfo(np.uint32).max:
                        df[col] = series.astype(np.uint32, copy=False)
                    else:
                        df[col] = series.astype(np.uint64, copy=False)
                else:
                    if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                        df[col] = series.astype(np.int8, copy=False)
                    elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                        df[col] = series.astype(np.int16, copy=False)
                    elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                        df[col] = series.astype(np.int32, copy=False)
                    else:
                        df[col] = series.astype(np.int64, copy=False)
            continue
        
        # Enforce compact float storage for memory budget
        if pd.api.types.is_float_dtype(col_type):
            df[col] = series.astype(np.float32, copy=False)
            # Use sparse storage to protect RAM on high-null columns
            if null_ratio >= 0.7:
                df[col] = pd.arrays.SparseArray(df[col], fill_value=np.nan)
            continue
    
    end_mem = df.memory_usage(deep=True).sum() / 1024**2
    
    if verbose:
        reduction = 100 * (start_mem - end_mem) / start_mem
        print(f'Memory usage decreased from {start_mem:.2f} MB to {end_mem:.2f} MB '
              f'({reduction:.1f}% reduction)')
    
    return df


def load_and_optimize_data(
    train_transaction_path: str,
    train_identity_path: str,
    test_transaction_path: str,
    test_identity_path: str
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load, optimize, and merge IEEE-CIS Fraud Detection datasets.

    Args:
        train_transaction_path: Path to train transaction CSV.
        train_identity_path: Path to train identity CSV.
        test_transaction_path: Path to test transaction CSV.
        test_identity_path: Path to test identity CSV.

    Returns:
        Tuple of (train_df, test_df).
    """
    print("=" * 80)
    print("SENTINEL FRAUD DETECTION - DATA INGESTION PIPELINE")
    print("=" * 80)
    
    # Enforce deterministic ingestion order for reproducibility
    print("\n[1/6] Loading train_transaction.csv...")
    try:
        train_transaction = pd.read_csv(train_transaction_path)
        print(f"✓ Loaded {train_transaction.shape[0]:,} transactions with "
              f"{train_transaction.shape[1]} features")
    except FileNotFoundError:
        raise FileNotFoundError(f"Training transaction file not found: {train_transaction_path}")
    except Exception as e:
        raise Exception(f"Error loading training transactions: {str(e)}")
    
    # Enforce memory budget immediately after load
    print("  → Optimizing memory...")
    train_transaction = reduce_mem_usage(train_transaction, verbose=True)
    gc.collect()
    
    # Preserve identity coverage without inflating peak RAM
    print("\n[2/6] Loading train_identity.csv...")
    try:
        train_identity = pd.read_csv(train_identity_path)
        print(f"✓ Loaded {train_identity.shape[0]:,} identity records with "
              f"{train_identity.shape[1]} features")
    except FileNotFoundError:
        raise FileNotFoundError(f"Training identity file not found: {train_identity_path}")
    except Exception as e:
        raise Exception(f"Error loading training identity: {str(e)}")
    
    print("  → Optimizing memory...")
    train_identity = reduce_mem_usage(train_identity, verbose=True)
    gc.collect()
    
    # Preserve base population via left join for fraud coverage
    print("\n[3/6] Merging training datasets...")
    print("  → Strategy: LEFT JOIN on TransactionID")
    print("  → Rationale: Preserve ALL transactions (base population)")
    print("    - Not all transactions have identity information")
    print("    - Missing identity data is a fraud signal itself")
    print("    - INNER JOIN would lose ~40% of transactions")
    
    train_df = train_transaction.merge(
        train_identity,
        on='TransactionID',
        how='left'
    )
    
    print(f"✓ Merged shape: {train_df.shape}")
    print(f"  Transactions retained: {train_df.shape[0]:,} / {train_transaction.shape[0]:,}")
    
    # Enforce memory budget via explicit cleanup
    del train_transaction, train_identity
    gc.collect()
    
    # Stage test ingestion with the same memory discipline
    print("\n[4/6] Loading test_transaction.csv...")
    try:
        test_transaction = pd.read_csv(test_transaction_path)
        print(f"✓ Loaded {test_transaction.shape[0]:,} transactions with "
              f"{test_transaction.shape[1]} features")
    except FileNotFoundError:
        raise FileNotFoundError(f"Test transaction file not found: {test_transaction_path}")
    except Exception as e:
        raise Exception(f"Error loading test transactions: {str(e)}")
    
    print("  → Optimizing memory...")
    test_transaction = reduce_mem_usage(test_transaction, verbose=True)
    gc.collect()
    
    # Preserve identity coverage while controlling memory
    print("\n[5/6] Loading test_identity.csv...")
    try:
        test_identity = pd.read_csv(test_identity_path)
        print(f"✓ Loaded {test_identity.shape[0]:,} identity records with "
              f"{test_identity.shape[1]} features")
    except FileNotFoundError:
        raise FileNotFoundError(f"Test identity file not found: {test_identity_path}")
    except Exception as e:
        raise Exception(f"Error loading test identity: {str(e)}")
    
    print("  → Optimizing memory...")
    test_identity = reduce_mem_usage(test_identity, verbose=True)
    gc.collect()
    
    # Preserve base population via left join for evaluation parity
    print("\n[6/6] Merging test datasets...")
    test_df = test_transaction.merge(
        test_identity,
        on='TransactionID',
        how='left'
    )
    
    print(f"✓ Merged shape: {test_df.shape}")
    
    # Enforce memory budget via explicit cleanup
    del test_transaction, test_identity
    gc.collect()
    
    # Enforce time-series integrity to prevent leakage
    print("\n[CRITICAL] Sorting training data chronologically...")
    print("  → Sorting by TransactionDT (time delta from reference point)")
    print("  → Purpose: Prevent FUTURE DATA LEAKAGE in time-series CV")
    print("  → Impact: Ensures train/validation splits respect temporal order")
    
    train_df = train_df.sort_values('TransactionDT').reset_index(drop=True)
    
    # Emit ingestion summary for auditability
    print("\n" + "=" * 80)
    print("INGESTION COMPLETE - FINAL SUMMARY")
    print("=" * 80)
    print(f"\nTraining Set:")
    print(f"  Shape: {train_df.shape}")
    print(f"  Memory: {train_df.memory_usage().sum() / 1024**2:.2f} MB")
    print(f"  Target Distribution: {train_df['isFraud'].value_counts().to_dict()}")
    print(f"  Fraud Rate: {train_df['isFraud'].mean() * 100:.2f}%")
    
    print(f"\nTest Set:")
    print(f"  Shape: {test_df.shape}")
    print(f"  Memory: {test_df.memory_usage().sum() / 1024**2:.2f} MB")
    
    print("\n✓ Pipeline execution successful")
    print("=" * 80)
    
    return train_df, test_df


if __name__ == "__main__":
    # Centralize pipeline configuration
    RAW_DATA_DIR = "./data/ieee-fraud-detection/"
    PROCESSED_DATA_DIR = "./data/"
    
    TRAIN_TRANSACTION = f"{RAW_DATA_DIR}train_transaction.csv"
    TRAIN_IDENTITY = f"{RAW_DATA_DIR}train_identity.csv"
    TEST_TRANSACTION = f"{RAW_DATA_DIR}test_transaction.csv"
    TEST_IDENTITY = f"{RAW_DATA_DIR}test_identity.csv"
    
    # Execute ingestion pipeline
    train_df, test_df = load_and_optimize_data(
        TRAIN_TRANSACTION,
        TRAIN_IDENTITY,
        TEST_TRANSACTION,
        TEST_IDENTITY
    )
    
    # Persist optimized datasets for reuse
    print("\n[OPTIONAL] Saving optimized datasets...")
    train_df.to_pickle(f"{PROCESSED_DATA_DIR}train_optimized.pkl")
    test_df.to_pickle(f"{PROCESSED_DATA_DIR}test_optimized.pkl")
    print("✓ Saved as pickle files for fast reloading")

import gc
import time
import pandas as pd
import numpy as np
from typing import Tuple
import warnings

warnings.filterwarnings('ignore')


def check_memory_usage(df: pd.DataFrame, stage: str = "") -> None:
    """
    Print current DataFrame memory usage.
    
    Args:
        df: DataFrame to check
        stage: Description of current processing stage
    """
    mem_usage = df.memory_usage(deep=True).sum() / 1024**2
    print(f"{'[' + stage + ']' if stage else 'Memory'} Usage: {mem_usage:.2f} MB")


def reduce_mem_usage(df: pd.DataFrame, float_downcast: bool = True) -> pd.DataFrame:
    """
    Downcast numeric types to reduce memory footprint.
    
    Args:
        df: Input DataFrame
        float_downcast: Whether to downcast float64 to float32
    
    Returns:
        Memory-optimized DataFrame
    """
    start_mem = df.memory_usage().sum() / 1024**2
    
    for col in df.columns:
        col_type = df[col].dtype
        
        if col_type != object:
            if pd.api.types.is_categorical_dtype(col_type):
                continue
            c_min = df[col].min()
            c_max = df[col].max()
            
            # Enforce compact integer storage for memory budget
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
            
            # Enforce compact float storage for memory budget
            elif float_downcast and str(col_type)[:5] == 'float':
                df[col] = df[col].astype(np.float32)
    
    end_mem = df.memory_usage().sum() / 1024**2
    reduction = 100 * (start_mem - end_mem) / start_mem
    print(f"  → Memory reduced: {start_mem:.2f} MB → {end_mem:.2f} MB ({reduction:.1f}% reduction)")
    
    return df


def create_uid(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create unique identifier by concatenating card1, addr1, and D1.
    
    Engineering Rationale:
    - card1: Primary card identifier
    - addr1: Billing address (geographic signal)
    - D1: Time delta feature (behavioral pattern)
    - Combination creates a pseudo-user identifier for velocity tracking
    
    Args:
        df: Input DataFrame
    
    Returns:
        DataFrame with 'uid' column
    """
    print("\n[1/5] Creating Unique Identifier (uid)...")
    
    # Standardize UID components to preserve grouping fidelity
    df['uid'] = (
        df['card1'].astype(str) + '_' +
        df['addr1'].astype(str) + '_' +
        df['D1'].astype(str)
    )
    
    print(f"  ✓ Created {df['uid'].nunique():,} unique identifiers")
    
    return df


def engineer_velocity_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create time-series velocity features using efficient rolling windows.
    
    Engineering Approach:
    - TransactionDT is in seconds (Unix timestamp delta)
    - 24 hours = 86,400 seconds
    - 30 days = 2,592,000 seconds
    
    Memory Optimization:
    - Use groupby().rolling() with custom time-based window
    - Avoid creating intermediate datetime columns
    - Process in-place where possible
    
    Args:
        df: Input DataFrame (must be sorted by TransactionDT)
    
    Returns:
        DataFrame with velocity features
    """
    print("\n[2/5] Engineering Velocity Features...")
    
    # Enforce chronological sorting for rolling windows
    if not df['TransactionDT'].is_monotonic_increasing:
        print("  ⚠ WARNING: Data not sorted by TransactionDT. Sorting now...")
        df = df.sort_values('TransactionDT').reset_index(drop=True)
    
    # Enforce velocity signal to capture bursty fraud behavior
    print("  → Feature A: 24h transaction frequency per uid...")

    # Anchor rolling lookback to preserve time-series integrity
    df['_lookback_24h'] = df['TransactionDT'] - 86400

    def count_transactions_24h(group):
        """Count transactions in 24h window for each row in group."""
        result = []
        times = group['TransactionDT'].values
        group_size = len(times)
        start_time = time.perf_counter()
        
        for i, current_time in enumerate(times):
            window_start = current_time - 86400
            count = np.sum((times >= window_start) & (times <= current_time))
            result.append(count)
            
            if group_size >= 50000 and ((i + 1) % 50000 == 0 or (i + 1) == group_size):
                elapsed = time.perf_counter() - start_time
                pct = ((i + 1) / group_size) * 100
                rate = (i + 1) / elapsed if elapsed > 0 else 0
                remaining = group_size - (i + 1)
                etc = remaining / rate if rate > 0 else 0
                print(f"  [Velocity 24h] {i + 1:,}/{group_size:,} ({pct:.1f}%) - ETC: {etc:.1f}s")
        
        return pd.Series(result, index=group.index)
    
    # Enforce per-uid aggregation to preserve behavioral signal
    df['uid_TransactionFreq_24h'] = df.groupby('uid', group_keys=False).apply(
        count_transactions_24h
    )
    
    print(f"    ✓ Mean 24h frequency: {df['uid_TransactionFreq_24h'].mean():.2f}")
    
    # Reclaim temporary columns to protect memory budget
    df.drop('_lookback_24h', axis=1, inplace=True)
    gc.collect()

    # Enforce long-horizon baseline to capture spending shifts
    print("  → Feature B: 30-day rolling mean transaction amount...")

    # Anchor rolling lookback to preserve time-series integrity
    df['_lookback_30d'] = df['TransactionDT'] - 2592000
    
    def rolling_mean_30d(group):
        """Calculate rolling mean of TransactionAmt over 30 days."""
        result = []
        times = group['TransactionDT'].values
        amounts = group['TransactionAmt'].values
        group_size = len(times)
        start_time = time.perf_counter()
        
        for i, current_time in enumerate(times):
            window_start = current_time - 2592000
            mask = (times >= window_start) & (times <= current_time)
            window_amounts = amounts[mask]

            if len(window_amounts) > 0:
                result.append(np.mean(window_amounts))
            else:
                result.append(np.nan)
            
            if group_size >= 50000 and ((i + 1) % 50000 == 0 or (i + 1) == group_size):
                elapsed = time.perf_counter() - start_time
                pct = ((i + 1) / group_size) * 100
                rate = (i + 1) / elapsed if elapsed > 0 else 0
                remaining = group_size - (i + 1)
                etc = remaining / rate if rate > 0 else 0
                print(f"  [Velocity 30d] {i + 1:,}/{group_size:,} ({pct:.1f}%) - ETC: {etc:.1f}s")
        
        return pd.Series(result, index=group.index)
    
    df['uid_TransactionAmt_mean_30d'] = df.groupby('uid', group_keys=False).apply(
        rolling_mean_30d
    )
    
    print(f"    ✓ Mean 30d avg amount: ${df['uid_TransactionAmt_mean_30d'].mean():.2f}")
    
    # Reclaim temporary columns to protect memory budget
    df.drop('_lookback_30d', axis=1, inplace=True)
    gc.collect()

    # Enforce compact feature storage for memory budget
    df['uid_TransactionFreq_24h'] = df['uid_TransactionFreq_24h'].astype(np.int16)
    df['uid_TransactionAmt_mean_30d'] = df['uid_TransactionAmt_mean_30d'].astype(np.float32)
    
    check_memory_usage(df, "After Velocity Features")
    
    return df


def engineer_divergence_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create divergence features comparing current behavior to historical patterns.
    
    Engineering Logic:
    - High ratio = current transaction much larger than typical (fraud signal)
    - Low ratio = current transaction much smaller than typical
    - NaN handling: Replace with 1.0 (no divergence) when no history exists
    
    Args:
        df: Input DataFrame with velocity features
    
    Returns:
        DataFrame with divergence features
    """
    print("\n[3/5] Engineering Divergence Features...")
    
    # Enforce deviation signal to surface anomalous spend
    print("  → Creating Amt_to_Mean_Ratio...")

    df['Amt_to_Mean_Ratio'] = df['TransactionAmt'] / df['uid_TransactionAmt_mean_30d']

    # Normalize missing or infinite ratios to a neutral baseline
    df['Amt_to_Mean_Ratio'] = df['Amt_to_Mean_Ratio'].replace([np.inf, -np.inf], np.nan)
    df['Amt_to_Mean_Ratio'] = df['Amt_to_Mean_Ratio'].fillna(1.0)

    # Enforce compact storage for downstream scoring
    df['Amt_to_Mean_Ratio'] = df['Amt_to_Mean_Ratio'].astype(np.float32)
    
    print(f"    ✓ Mean ratio: {df['Amt_to_Mean_Ratio'].mean():.2f}")
    print(f"    ✓ Ratio > 3x (anomaly): {(df['Amt_to_Mean_Ratio'] > 3).sum():,} transactions")
    
    check_memory_usage(df, "After Divergence Features")
    
    return df


def engineer_frequency_encoding(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply frequency encoding to categorical features.
    
    Engineering Rationale:
    - Frequency encoding captures the prevalence of a value
    - Rare values (low frequency) are often fraud signals
    - More memory-efficient than one-hot encoding
    - Preserves ordinality based on occurrence
    
    Args:
        df: Input DataFrame
    
    Returns:
        DataFrame with frequency-encoded features
    """
    print("\n[4/5] Applying Frequency Encoding...")
    
    categorical_cols = ['card1', 'P_emaildomain']
    
    for col in categorical_cols:
        if col in df.columns:
            print(f"  → Encoding {col}...")

            # Encode rarity to strengthen fraud signal
            freq_map = df[col].value_counts().to_dict()

            new_col_name = f'{col}_freq'
            df[new_col_name] = df[col].map(freq_map)

            # Normalize missing values for model stability
            df[new_col_name] = df[new_col_name].fillna(0).astype(np.int32)
            
            print(f"    ✓ Unique values: {df[col].nunique():,}")
            print(f"    ✓ Mean frequency: {df[new_col_name].mean():.2f}")
    
    check_memory_usage(df, "After Frequency Encoding")
    
    return df


def optimize_final_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Final memory optimization pass.
    
    Args:
        df: Input DataFrame
    
    Returns:
        Optimized DataFrame
    """
    print("\n[5/5] Final Memory Optimization...")

    # Enforce global memory budget before persistence
    df = reduce_mem_usage(df, float_downcast=True)
    
    gc.collect()
    
    return df


def run_feature_engineering_pipeline(df: pd.DataFrame) -> pd.DataFrame:
    """
    Execute complete feature engineering pipeline.
    
    Pipeline Steps:
    1. Create unique identifier (uid)
    2. Engineer velocity features (24h frequency, 30d mean)
    3. Engineer divergence features (amount ratios)
    4. Apply frequency encoding (categorical features)
    5. Final memory optimization
    
    Args:
        df: Input DataFrame (must contain required columns)
    
    Returns:
        Feature-engineered DataFrame
    """
    print("=" * 80)
    print("SENTINEL - FEATURE ENGINEERING PIPELINE")
    print("=" * 80)
    
    # Enforce required schema for production features
    required_cols = ['TransactionDT', 'TransactionAmt', 'card1', 'addr1', 'D1']
    missing_cols = [col for col in required_cols if col not in df.columns]
    
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    print(f"\nInput Shape: {df.shape}")
    check_memory_usage(df, "Initial")
    
    # Enforce time-series integrity for velocity features
    print("\n[Pre-Processing] Ensuring chronological order...")
    if not df['TransactionDT'].is_monotonic_increasing:
        df = df.sort_values('TransactionDT').reset_index(drop=True)
        print("  ✓ Sorted by TransactionDT")
    else:
        print("  ✓ Already sorted")
    
    # Execute Sentinel feature pipeline stages
    df = create_uid(df)
    df = engineer_velocity_features(df)
    df = engineer_divergence_features(df)
    df = engineer_frequency_encoding(df)
    df = optimize_final_dataframe(df)
    
    # Emit pipeline summary for auditability
    print("\n" + "=" * 80)
    print("FEATURE ENGINEERING COMPLETE")
    print("=" * 80)
    print(f"\nFinal Shape: {df.shape}")
    print(f"New Features Created: {df.shape[1] - len(required_cols)}")
    check_memory_usage(df, "Final")
    
    print("\n✓ Pipeline execution successful")
    print("=" * 80)
    
    return df


if __name__ == "__main__":
    # Signal module usage for local runs
    print("\n[INFO] This is a module script. Import and use run_feature_engineering_pipeline()")
    print("\nExample:")
    print("  from feature_engineering import run_feature_engineering_pipeline")
    print("  train_df = pd.read_pickle(\1../data/processed/train_optimized.pkl')")
    print("  train_df = run_feature_engineering_pipeline(train_df)")
    
    # Enable local demo for validation
    print("\n[DEMO] Creating sample dataset for demonstration...")
    
    np.random.seed(42)
    n_samples = 10000
    
    sample_df = pd.DataFrame({
        'TransactionDT': np.sort(np.random.randint(0, 15000000, n_samples)),
        'TransactionAmt': np.random.exponential(100, n_samples),
        'card1': np.random.randint(1000, 2000, n_samples),
        'addr1': np.random.randint(100, 500, n_samples),
        'D1': np.random.randint(0, 100, n_samples),
        'P_emaildomain': np.random.choice(['gmail.com', 'yahoo.com', 'hotmail.com', None], n_samples)
    })
    
    print(f"\nSample data created: {sample_df.shape}")
    
    # Execute pipeline on sample data
    sample_df = run_feature_engineering_pipeline(sample_df)
    
    print("\n[DEMO] Sample output:")
    print(sample_df[['TransactionAmt', 'uid_TransactionFreq_24h', 
                     'uid_TransactionAmt_mean_30d', 'Amt_to_Mean_Ratio']].head(10))

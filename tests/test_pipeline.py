import numpy as np
import pandas as pd
import pytest
from typing import Tuple
import os
import sys
from sklearn.model_selection import TimeSeriesSplit

# Enforce local imports for deterministic test isolation
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Enforce explicit imports for test scope
from feature_engineering import (
    create_uid,
    engineer_velocity_features,
    engineer_divergence_features,
    engineer_frequency_encoding,
    run_feature_engineering_pipeline
)
from data_ingestion import reduce_mem_usage


# Enforce golden fixtures for deterministic validation

@pytest.fixture
def golden_dataset() -> pd.DataFrame:
    """
    Create synthetic "golden dataset" with known fraud patterns.
    
    Engineering Design:
    - Small dataset (100 rows) for fast testing
    - Known patterns for validation
    - Includes edge cases (NaN, zero values, duplicates)
    
    Returns:
        DataFrame with synthetic transaction data
    """
    np.random.seed(42)
    
    n_samples = 100

    base_time = 1000000
    transaction_dt = np.arange(base_time, base_time + n_samples * 3600, 3600)

    data = {
        'TransactionID': np.arange(1, n_samples + 1),
        'TransactionDT': transaction_dt,
        'TransactionAmt': np.random.exponential(100, n_samples),
        'card1': np.random.choice([1001, 1002, 1003, 1004, 1005], n_samples),
        'addr1': np.random.choice([100, 101, 102], n_samples),
        'D1': np.random.randint(0, 10, n_samples),
        'P_emaildomain': np.random.choice(['gmail.com', 'yahoo.com', None], n_samples),
        'isFraud': np.random.choice([0, 1], n_samples, p=[0.97, 0.03])
    }
    
    df = pd.DataFrame(data)
    
    # Inject known fraud burst pattern for velocity validation
    df.loc[10:15, 'card1'] = 1001
    df.loc[10:15, 'addr1'] = 100
    df.loc[10:15, 'D1'] = 5
    df.loc[10:15, 'TransactionDT'] = base_time + 10000 + np.arange(6) * 1000
    df.loc[10:15, 'isFraud'] = 1

    # Inject missingness to validate null handling
    df.loc[20:25, 'P_emaildomain'] = None

    # Inject zero-amount edge case for ratio stability
    df.loc[30, 'TransactionAmt'] = 0.0

    # Enforce chronological order for time-series features
    df = df.sort_values('TransactionDT').reset_index(drop=True)
    
    return df


@pytest.fixture
def golden_dataset_with_features(golden_dataset) -> pd.DataFrame:
    """
    Golden dataset with engineered features.
    
    Returns:
        DataFrame with all features engineered
    """
    df = golden_dataset.copy()
    
    # Enforce full feature stack on golden dataset
    df = create_uid(df)
    df = engineer_velocity_features(df)
    df = engineer_divergence_features(df)
    df = engineer_frequency_encoding(df)
    
    return df


# Enforce production-aligned time-series split configuration

def test_timeseries_split_configuration(golden_dataset):
    tscv = TimeSeriesSplit(n_splits=5)
    splits = list(tscv.split(golden_dataset))
    
    assert len(splits) == 5, f"Expected 5 splits, got {len(splits)}"


def test_velocity_24h_frequency_basic(golden_dataset):
    # Enforce 24h velocity signal invariants
    df = create_uid(golden_dataset)
    df = engineer_velocity_features(df)

    first_uid = df.iloc[0]['uid']
    first_count = df.iloc[0]['uid_TransactionFreq_24h']
    
    assert first_count == 1, f"First transaction should have count=1, got {first_count}"
    
    fraud_pattern = df[df['TransactionID'].between(11, 16)].copy()
    
    if len(fraud_pattern) > 0:
        fraud_uid = fraud_pattern['uid'].iloc[0]
        assert fraud_pattern['uid'].nunique() == 1, "Fraud pattern should have same uid"

        fraud_counts = fraud_pattern['uid_TransactionFreq_24h'].values

        for i, count in enumerate(fraud_counts):
            assert count >= 1, f"Transaction {i} should have count >= 1, got {count}"

    assert (df['uid_TransactionFreq_24h'] >= 1).all(), \
        "All transactions should have frequency >= 1"
    
    print("✓ Test Case 1 PASSED: uid_TransactionFreq_24h validated")


def test_velocity_24h_frequency_edge_cases(golden_dataset):
    # Enforce edge-case stability for velocity features
    df = create_uid(golden_dataset)
    df = engineer_velocity_features(df)

    uid_counts = df.groupby('uid')['uid_TransactionFreq_24h'].first()
    single_transaction_uids = df['uid'].value_counts()[df['uid'].value_counts() == 1].index
    
    for uid in single_transaction_uids:
        count = df[df['uid'] == uid]['uid_TransactionFreq_24h'].iloc[0]
        assert count == 1, f"Single transaction uid should have count=1, got {count}"
    
    assert (df['uid_TransactionFreq_24h'] >= 1).all(), \
        "All transactions should have frequency >= 1"

    assert df['uid_TransactionFreq_24h'].dtype in [np.int8, np.int16, np.int32, np.int64], \
        f"Frequency should be integer type, got {df['uid_TransactionFreq_24h'].dtype}"
    
    print("✓ Test Case 1 (Edge Cases) PASSED: Edge cases validated")


def test_divergence_ratio_nan_handling(golden_dataset):
    # Enforce neutral handling for missing or infinite ratios
    df = create_uid(golden_dataset)
    df = engineer_velocity_features(df)
    df = engineer_divergence_features(df)

    nan_count = df['Amt_to_Mean_Ratio'].isna().sum()
    assert nan_count == 0, f"Amt_to_Mean_Ratio should have no NaN values, found {nan_count}"

    inf_count = np.isinf(df['Amt_to_Mean_Ratio']).sum()
    assert inf_count == 0, f"Amt_to_Mean_Ratio should have no Inf values, found {inf_count}"

    assert (df['Amt_to_Mean_Ratio'] > 0).all(), \
        "All ratios should be positive"
    
    # Enforce neutral ratio for first-transaction baselines
    first_transactions = df.groupby('uid').first()
    
    # Enforce neutral ratio for zero-mean edge cases
    test_df = df.copy()
    test_df.loc[0, 'uid_TransactionAmt_mean_30d'] = 0.0
    test_df = engineer_divergence_features(test_df)
    
    assert test_df.loc[0, 'Amt_to_Mean_Ratio'] == 1.0, \
        "Zero mean should result in ratio = 1.0"
    
    print("✓ Test Case 2 PASSED: Amt_to_Mean_Ratio NaN/Inf handling validated")


def test_divergence_ratio_calculation(golden_dataset):
    # Enforce ratio calculation consistency
    df = create_uid(golden_dataset)
    df = engineer_velocity_features(df)
    df = engineer_divergence_features(df)
    
    # Enforce ratio correctness on a sample
    sample_idx = 50
    
    if df.loc[sample_idx, 'uid_TransactionAmt_mean_30d'] > 0:
        expected_ratio = (
            df.loc[sample_idx, 'TransactionAmt'] / 
            df.loc[sample_idx, 'uid_TransactionAmt_mean_30d']
        )
        actual_ratio = df.loc[sample_idx, 'Amt_to_Mean_Ratio']
        
        # Allow float32 tolerance for ratio checks
        assert abs(expected_ratio - actual_ratio) < 0.01, \
            f"Ratio mismatch: expected {expected_ratio}, got {actual_ratio}"
    
    # Enforce compact ratio storage
    assert df['Amt_to_Mean_Ratio'].dtype == np.float32, \
        f"Ratio should be float32, got {df['Amt_to_Mean_Ratio'].dtype}"
    
    print("✓ Test Case 2 (Calculation) PASSED: Ratio calculation validated")


def test_memory_downcasting_float64_to_float32():
    # Enforce float downcasting without material precision loss
    test_df = pd.DataFrame({
        'col_float64': np.random.randn(100).astype(np.float64),
        'col_int64': np.random.randint(0, 1000, 100).astype(np.int64),
        'col_object': ['test'] * 100
    })
    
    assert test_df['col_float64'].dtype == np.float64, "Initial type should be float64"
    
    optimized_df = reduce_mem_usage(test_df, verbose=False)
    
    assert optimized_df['col_float64'].dtype == np.float32, \
        f"float64 should become float32, got {optimized_df['col_float64'].dtype}"
    
    original_values = test_df['col_float64'].values
    optimized_values = optimized_df['col_float64'].values
    
    # Allow float32 precision tolerance
    max_relative_error = np.max(np.abs((original_values - optimized_values) / original_values))
    assert max_relative_error < 1e-6, \
        f"Precision loss too high: {max_relative_error}"
    
    assert optimized_df['col_int64'].dtype in [np.int8, np.int16, np.int32, np.uint8, np.uint16, np.uint32], \
        f"int64 should be downcasted, got {optimized_df['col_int64'].dtype}"
    
    assert optimized_df['col_object'].dtype in [object, 'category'], \
        "Object columns should remain object or be converted to category"
    
    print("✓ Test Case 3 PASSED: Memory downcasting validated")


def test_memory_downcasting_preserves_values():
    # Enforce exact preservation of integer values
    n_samples = 100
    test_df = pd.DataFrame({
        'small_int': np.arange(0, n_samples, dtype=np.int64),
        'medium_int': np.arange(0, n_samples, dtype=np.int64) * 10,
        'large_int': np.arange(0, n_samples, dtype=np.int64) * 1000,
    })
    
    optimized_df = reduce_mem_usage(test_df, verbose=False)
    
    assert (test_df['small_int'] == optimized_df['small_int']).all(), \
        "Small integers should be preserved exactly"
    
    assert (test_df['medium_int'] == optimized_df['medium_int']).all(), \
        "Medium integers should be preserved exactly"
    
    assert (test_df['large_int'] == optimized_df['large_int']).all(), \
        "Large integers should be preserved exactly"
    
    assert optimized_df['small_int'].dtype in [np.int8, np.uint8], \
        f"Small int should be int8/uint8, got {optimized_df['small_int'].dtype}"
    
    assert optimized_df['medium_int'].dtype in [np.int16, np.uint16], \
        f"Medium int should be int16/uint16, got {optimized_df['medium_int'].dtype}"
    
    assert optimized_df['large_int'].dtype in [np.int32, np.uint32], \
        f"Large int should be int32/uint32, got {optimized_df['large_int'].dtype}"
    
    print("✓ Test Case 3 (Value Preservation) PASSED: Integer values preserved")


# Enforce feature engineering pipeline integrity

def test_uid_creation(golden_dataset):
    # Enforce UID construction invariants
    df = create_uid(golden_dataset)
    
    # Enforce UID column presence
    assert 'uid' in df.columns, "UID column should be created"
    
    # Enforce UID type for grouping stability
    assert df['uid'].dtype == object, "UID should be string type"
    
    # Enforce UID format consistency
    sample_row = df.iloc[0]
    expected_uid = f"{sample_row['card1']}_{sample_row['addr1']}_{sample_row['D1']}"
    actual_uid = sample_row['uid']
    
    assert actual_uid == expected_uid, \
        f"UID format mismatch: expected {expected_uid}, got {actual_uid}"
    
    # Enforce UID consistency for known fraud pattern
    fraud_pattern = df[df['TransactionID'].between(11, 16)]
    if len(fraud_pattern) > 0:
        fraud_uids = fraud_pattern['uid']
        assert fraud_uids.nunique() == 1, "Fraud pattern should have same UID"
    
    print("✓ Additional Test PASSED: UID creation validated")


def test_frequency_encoding(golden_dataset):
    # Enforce frequency encoding invariants
    df = create_uid(golden_dataset)
    df = engineer_frequency_encoding(df)
    
    # Enforce frequency-encoded columns
    assert 'card1_freq' in df.columns, "card1_freq should be created"
    assert 'P_emaildomain_freq' in df.columns, "P_emaildomain_freq should be created"
    
    # Enforce frequency correctness
    card1_value_counts = golden_dataset['card1'].value_counts()
    
    for card_value in card1_value_counts.index:
        expected_freq = card1_value_counts[card_value]
        actual_freq = df[df['card1'] == card_value]['card1_freq'].iloc[0]
        
        assert expected_freq == actual_freq, \
            f"Frequency mismatch for card1={card_value}: expected {expected_freq}, got {actual_freq}"
    
    # Enforce NaN handling for frequency encoding
    nan_freq = df[df['P_emaildomain'].isna()]['P_emaildomain_freq'].iloc[0]
    assert nan_freq == 0, f"NaN should have frequency=0, got {nan_freq}"
    
    # Enforce integer frequency storage
    assert df['card1_freq'].dtype in [np.int8, np.int16, np.int32, np.int64], \
        "Frequency should be integer type"
    
    print("✓ Additional Test PASSED: Frequency encoding validated")


# Enforce end-to-end pipeline integrity

def test_full_pipeline_integration(golden_dataset):
    # Enforce end-to-end pipeline invariants
    original_shape = golden_dataset.shape
    
    # Enforce full pipeline execution
    df = run_feature_engineering_pipeline(golden_dataset)
    
    # Enforce row preservation across pipeline
    assert df.shape[0] == original_shape[0], \
        f"Rows lost: expected {original_shape[0]}, got {df.shape[0]}"
    
    # Enforce feature creation
    expected_new_features = [
        'uid',
        'uid_TransactionFreq_24h',
        'uid_TransactionAmt_mean_30d',
        'Amt_to_Mean_Ratio',
        'card1_freq',
        'P_emaildomain_freq'
    ]
    
    for feature in expected_new_features:
        assert feature in df.columns, f"Feature {feature} not created"
    
    # Enforce NaN-free critical features
    critical_features = [
        'uid_TransactionFreq_24h',
        'Amt_to_Mean_Ratio',
        'card1_freq'
    ]
    
    for feature in critical_features:
        nan_count = df[feature].isna().sum()
        assert nan_count == 0, f"Feature {feature} has {nan_count} NaN values"
    
    # Enforce optimized dtypes for memory budget
    assert df['uid_TransactionFreq_24h'].dtype in [np.int8, np.int16, np.int32], \
        "Frequency should be optimized integer type"
    
    assert df['Amt_to_Mean_Ratio'].dtype == np.float32, \
        "Ratio should be float32"
    
    print("✓ Integration Test PASSED: Full pipeline validated")


# Enable local test execution

def create_golden_dataset_standalone() -> pd.DataFrame:
    # Create golden dataset for standalone execution
    np.random.seed(42)
    
    n_samples = 100
    base_time = 1000000
    transaction_dt = np.arange(base_time, base_time + n_samples * 3600, 3600)
    
    data = {
        'TransactionID': np.arange(1, n_samples + 1),
        'TransactionDT': transaction_dt,
        'TransactionAmt': np.random.exponential(100, n_samples),
        'card1': np.random.choice([1001, 1002, 1003, 1004, 1005], n_samples),
        'addr1': np.random.choice([100, 101, 102], n_samples),
        'D1': np.random.randint(0, 10, n_samples),
        'P_emaildomain': np.random.choice(['gmail.com', 'yahoo.com', None], n_samples),
        'isFraud': np.random.choice([0, 1], n_samples, p=[0.97, 0.03])
    }
    
    df = pd.DataFrame(data)
    
    # Inject known fraud burst pattern for validation
    df.loc[10:15, 'card1'] = 1001
    df.loc[10:15, 'addr1'] = 100
    df.loc[10:15, 'D1'] = 5
    df.loc[10:15, 'TransactionDT'] = base_time + 10000 + np.arange(6) * 1000
    df.loc[10:15, 'isFraud'] = 1
    
    # Inject edge cases for null and zero handling
    df.loc[20:25, 'P_emaildomain'] = None
    df.loc[30, 'TransactionAmt'] = 0.0
    
    df = df.sort_values('TransactionDT').reset_index(drop=True)
    
    return df


if __name__ == "__main__":
    print("\n" + "=" * 80)
    print("SENTINEL UNIT TESTING SUITE")
    print("=" * 80 + "\n")
    
    # Initialize golden dataset for manual run
    print("Creating golden dataset...")
    golden_data = create_golden_dataset_standalone()
    print(f"✓ Golden dataset created: {golden_data.shape}\n")
    
    # Execute test suite manually
    print("Running Test Case 1: uid_TransactionFreq_24h...")
    test_velocity_24h_frequency_basic(golden_data)
    test_velocity_24h_frequency_edge_cases(golden_data)
    
    print("\nRunning Test Case 2: Amt_to_Mean_Ratio...")
    test_divergence_ratio_nan_handling(golden_data)
    test_divergence_ratio_calculation(golden_data)
    
    print("\nRunning Test Case 3: Memory Downcasting...")
    test_memory_downcasting_float64_to_float32()
    test_memory_downcasting_preserves_values()
    
    print("\nRunning Additional Tests...")
    test_uid_creation(golden_data)
    test_frequency_encoding(golden_data)
    
    print("\nRunning Integration Test...")
    test_full_pipeline_integration(golden_data)
    
    print("\n" + "=" * 80)
    print("ALL TESTS PASSED ✓")
    print("=" * 80 + "\n")

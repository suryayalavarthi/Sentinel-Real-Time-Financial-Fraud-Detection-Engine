# Sentinel - Feature Engineering Documentation

## 🎯 Overview

The Sentinel feature engineering pipeline creates **velocity** and **behavioral** features for fraud detection using memory-efficient Pandas operations optimized for datasets with 1M+ rows.

## 📋 Features Created

### 1. **Velocity Features** (Time-Series)

| Feature | Description | Engineering Logic | Memory Impact |
|---------|-------------|-------------------|---------------|
| `uid_TransactionFreq_24h` | Count of transactions per uid in last 24 hours | Rolling window on sorted data | int16 (2 bytes) |
| `uid_TransactionAmt_mean_30d` | Rolling mean of transaction amounts over 30 days | Time-based aggregation | float32 (4 bytes) |

**Key Implementation Details:**
- **Time Window Logic**: Uses `TransactionDT` (seconds) directly without datetime conversion
- **24 hours** = 86,400 seconds
- **30 days** = 2,592,000 seconds
- **Sorting Requirement**: Data MUST be sorted by `TransactionDT` before processing

### 2. **Divergence Features**

| Feature | Description | Formula | Fraud Signal |
|---------|-------------|---------|--------------|
| `Amt_to_Mean_Ratio` | Current amount vs. historical average | `TransactionAmt / uid_TransactionAmt_mean_30d` | Ratio > 3 = anomaly |

**Edge Case Handling:**
- Division by zero → Replaced with 1.0 (neutral)
- No historical data (NaN) → Replaced with 1.0
- Infinite values → Replaced with NaN, then 1.0

### 3. **Frequency Encoding**

| Feature | Description | Fraud Insight |
|---------|-------------|---------------|
| `card1_freq` | Number of times this card appears in dataset | Rare cards = higher fraud risk |
| `P_emaildomain_freq` | Number of times this email domain appears | Rare domains = higher fraud risk |

## 🧠 Engineering Rationale

### Why Custom Rolling Windows?

**Standard Approach (Avoided):**
```python
df['datetime'] = pd.to_datetime(df['TransactionDT'], unit='s')
df.set_index('datetime').rolling('24h').count()
```
**Problems:**
- Creates datetime column (8 bytes per row)
- Creates datetime index (duplicate memory)
- Total overhead: ~16 MB per 1M rows

**Our Approach (Optimized):**
```python
def count_transactions_24h(group):
    times = group['TransactionDT'].values  # NumPy array (fast)
    for current_time in times:
        window_start = current_time - 86400
        count = np.sum((times >= window_start) & (times <= current_time))
```
**Benefits:**
- No datetime conversion
- Vectorized NumPy operations within groups
- Memory overhead: ~0 MB (in-place calculation)

### Why LEFT JOIN in Data Ingestion?

```python
train_df = train_transaction.merge(train_identity, on='TransactionID', how='left')
```

**Rationale:**
- ~40% of transactions have NO identity data
- Missing identity is itself a fraud signal
- INNER JOIN would lose critical information
- Preserves the complete transaction population

## 🚀 Usage Examples

### Basic Usage

```python
from feature_engineering import run_feature_engineering_pipeline
import pandas as pd

# Load data
df = pd.read_pickle('data/train_optimized.pkl')

# Run pipeline
df = run_feature_engineering_pipeline(df)

# Inspect results
print(df[['TransactionAmt', 'uid_TransactionFreq_24h', 
          'Amt_to_Mean_Ratio']].describe())
```

### Complete Pipeline

```python
# Run everything
python run_pipeline.py
```

This executes:
1. Data ingestion (load + merge + optimize)
2. Feature engineering (velocity + divergence + encoding)
3. Save processed data as pickle files

### Memory Monitoring

```python
from feature_engineering import check_memory_usage

check_memory_usage(df, "After Feature Engineering")
# Output: [After Feature Engineering] Usage: 245.32 MB
```

## ⚙️ Performance Characteristics

### Time Complexity

| Operation | Complexity | Notes |
|-----------|-----------|-------|
| UID Creation | O(n) | String concatenation |
| 24h Frequency | O(n × m) | n = rows, m = avg transactions per uid |
| 30d Rolling Mean | O(n × m) | Same as above |
| Frequency Encoding | O(n) | Hash map lookup |

**Optimization:** Groupby operations are parallelizable (future enhancement)

### Memory Complexity

| Stage | Memory Usage (1M rows) | Peak Memory |
|-------|------------------------|-------------|
| Raw Data (float64) | ~800 MB | 800 MB |
| After Optimization | ~400 MB | 800 MB |
| After Feature Eng | ~450 MB | 900 MB |

**Peak Memory** occurs during rolling window calculations (temporary arrays).

## 🔍 Validation & Testing

### Verify Velocity Features

```python
# Check 24h frequency logic
sample_uid = df['uid'].iloc[0]
sample_time = df['TransactionDT'].iloc[0]

# Manual calculation
manual_count = df[
    (df['uid'] == sample_uid) & 
    (df['TransactionDT'] >= sample_time - 86400) &
    (df['TransactionDT'] <= sample_time)
].shape[0]

# Compare with feature
feature_count = df.iloc[0]['uid_TransactionFreq_24h']

assert manual_count == feature_count, "Velocity feature mismatch!"
```

### Verify Divergence Features

```python
# Check ratio calculation
row = df.iloc[100]
expected_ratio = row['TransactionAmt'] / row['uid_TransactionAmt_mean_30d']
actual_ratio = row['Amt_to_Mean_Ratio']

assert abs(expected_ratio - actual_ratio) < 0.01, "Ratio mismatch!"
```

## ⚠️ Common Pitfalls & Solutions

### 1. **Unsorted Data**

**Problem:** Rolling windows produce incorrect results if data isn't sorted.

**Solution:** Pipeline automatically checks and sorts:
```python
if not df['TransactionDT'].is_monotonic_increasing:
    df = df.sort_values('TransactionDT').reset_index(drop=True)
```

### 2. **Memory Overflow**

**Problem:** Large datasets cause OOM errors during rolling calculations.

**Solution:** Process in chunks (future enhancement):
```python
# Split by uid groups, process separately
for uid, group in df.groupby('uid'):
    group = engineer_velocity_features(group)
```

### 3. **NaN Propagation**

**Problem:** NaN values in source columns create NaN features.

**Solution:** Explicit NaN handling in each feature:
```python
df['Amt_to_Mean_Ratio'] = df['Amt_to_Mean_Ratio'].fillna(1.0)
```

## 📊 Feature Importance (Expected)

Based on fraud detection literature:

1. **uid_TransactionFreq_24h** (High) - Velocity is a strong fraud signal
2. **Amt_to_Mean_Ratio** (High) - Behavioral divergence indicates fraud
3. **card1_freq** (Medium) - Rare cards are riskier
4. **P_emaildomain_freq** (Low) - Weak signal, but useful in ensemble

## 🔮 Future Enhancements

1. **Parallel Processing**: Use `dask` or `modin` for multi-core groupby operations
2. **GPU Acceleration**: Implement rolling windows with `cuDF` (RAPIDS)
3. **Additional Features**:
   - Transaction frequency by hour-of-day
   - Geographic velocity (distance between consecutive transactions)
   - Device fingerprint frequency
4. **Automated Feature Selection**: Use SHAP values to identify top features

## 📚 References

- IEEE-CIS Fraud Detection Dataset: https://www.kaggle.com/c/ieee-fraud-detection
- Pandas Memory Optimization: https://pandas.pydata.org/docs/user_guide/scale.html
- Time-Series Feature Engineering: https://www.kaggle.com/c/ieee-fraud-detection/discussion/111284

---

**Last Updated**: 2026-01-31  
**Version**: 1.0.0

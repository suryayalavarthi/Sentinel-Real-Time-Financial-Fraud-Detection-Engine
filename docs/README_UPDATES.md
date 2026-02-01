# Sentinel MLOps Extensions - Setup & Usage Guide

## 🎯 Overview

This document covers the newly added MLOps capabilities:
1. **SHAP Interpretability** - Explain model decisions
2. **Automated Unit Testing** - Validate pipeline logic
3. **Production Monitoring** - Detect drift and performance degradation

---

## 📦 Installation

### 1. Install Additional Dependencies

```bash
# Core MLOps dependencies
pip install shap pytest scipy

# Optional: For better visualizations
pip install matplotlib seaborn
```

### 2. Update requirements.txt

The following dependencies have been added:

```python
# MLOps & Monitoring
shap>=0.40.0           # Model interpretability
pytest>=7.0.0          # Unit testing
scipy>=1.7.0           # Statistical tests for drift detection

# Visualization (Optional)
matplotlib>=3.4.0
seaborn>=0.11.0
```

---

## 🔍 SHAP Interpretability

### Purpose

SHAP (SHapley Additive exPlanations) provides transparent, actionable explanations for fraud detection decisions.

### Quick Start

```bash
python interpretability.py
```

### Expected Output

```
SENTINEL MODEL INTERPRETABILITY - SHAP INITIALIZATION
================================================================================

[1/4] Loading model from ./models/sentinel_fraud_model.pkl...
✓ Model loaded

[2/4] Loading data from ./data/train_engineered.pkl...
✓ Data loaded: (590540, 441)

[3/4] Preparing features...
✓ Features prepared: 437 features

[4/4] Creating SHAP background dataset (100 samples)...
✓ Background dataset created:
  - Fraud samples: 10
  - Legitimate samples: 90

✓ SHAP explainer initialized

GENERATING SHAP SUMMARY PLOT
================================================================================

[1/3] Sampling 1000 transactions...
[2/3] Computing SHAP values (this may take a few minutes)...
[3/3] Generating summary plot (top 20 features)...
✓ Saved: ./reports/shap_summary.png
```

### Generated Visualizations

1. **`shap_summary.png`** - Global feature importance
   - Shows which features matter most across all predictions
   - Color indicates feature value (red=high, blue=low)
   - Position indicates impact on prediction

2. **`fraud_explanation.png`** - Waterfall plot for fraud transaction
   - Shows how features contribute to fraud prediction
   - Cumulative impact visualization

3. **`legitimate_explanation.png`** - Waterfall plot for legitimate transaction
   - Shows why transaction was classified as legitimate

### Interpreting SHAP Plots

#### Summary Plot

```
Feature                        SHAP Value Impact
─────────────────────────────────────────────────
uid_TransactionFreq_24h        ████████████ (High)
Amt_to_Mean_Ratio              ██████████   (High)
TransactionAmt                 ████████     (Medium)
card1_freq                     ██████       (Medium)
P_emaildomain_freq             ████         (Low)
```

**Reading the plot:**
- **X-axis**: SHAP value (impact on prediction)
- **Y-axis**: Features (sorted by importance)
- **Color**: Feature value (red=high, blue=low)
- **Dots**: Individual transactions

**Example Insight:**
- High `uid_TransactionFreq_24h` (red dots on right) → Pushes prediction toward fraud
- Low `uid_TransactionFreq_24h` (blue dots on left) → Pushes prediction toward legitimate

#### Waterfall Plot

```
Base Value: 0.035 (3.5% fraud rate)

Feature                    Value      Impact      Cumulative
──────────────────────────────────────────────────────────────
uid_TransactionFreq_24h    15         +0.45       0.485
Amt_to_Mean_Ratio          5.2        +0.30       0.785
TransactionAmt             999.99     +0.10       0.885
card1_freq                 3          +0.05       0.935
...
Final Prediction: 0.935 (93.5% fraud probability)
```

**Reading the plot:**
- **Base Value**: Expected value (average fraud rate)
- **Each bar**: Feature contribution (positive = toward fraud, negative = toward legitimate)
- **Final Value**: Actual prediction

### Programmatic Usage

```python
from interpretability import SentinelExplainer

# Initialize explainer
explainer = SentinelExplainer(
    model_path="./models/sentinel_fraud_model.pkl",
    data_path="./data/train_engineered.pkl",
    background_size=100  # Smaller = faster, larger = more accurate
)

# Generate summary plot
explainer.generate_summary_plot(
    sample_size=1000,
    max_display=20,
    save_path="./reports/shap_summary.png"
)

# Explain specific transaction by index
explanation = explainer.explain_transaction(
    transaction_idx=42,
    plot_type='waterfall',  # or 'force'
    save_path="./reports/transaction_42.png"
)

# Explain by TransactionID (if available)
explanation = explainer.explain_transaction_by_id(
    transaction_id=2987123,
    plot_type='waterfall'
)

# Batch explanations
explanations = explainer.batch_explain(
    n_samples=10,
    output_dir="./reports/explanations"
)
```

### Performance Considerations

| Background Size | Initialization Time | Explanation Time | Accuracy |
|----------------|---------------------|------------------|----------|
| 50 | ~5 seconds | ~2 seconds/txn | Good |
| 100 | ~10 seconds | ~3 seconds/txn | Better |
| 500 | ~30 seconds | ~5 seconds/txn | Best |

**Recommendation**: Use 100-200 samples for production (good balance).

---

## 🧪 Automated Unit Testing

### Purpose

Validate critical business logic in the feature engineering pipeline using pytest.

### Quick Start

```bash
# Run all tests
pytest tests/test_pipeline.py -v

# Run specific test
pytest tests/test_pipeline.py::test_velocity_24h_frequency_basic -v

# Run with coverage
pytest tests/test_pipeline.py --cov=feature_engineering --cov-report=html
```

### Expected Output

```
tests/test_pipeline.py::test_velocity_24h_frequency_basic PASSED        [ 10%]
tests/test_pipeline.py::test_velocity_24h_frequency_edge_cases PASSED   [ 20%]
tests/test_pipeline.py::test_divergence_ratio_nan_handling PASSED       [ 30%]
tests/test_pipeline.py::test_divergence_ratio_calculation PASSED        [ 40%]
tests/test_pipeline.py::test_memory_downcasting_float64_to_float32 PASSED [ 50%]
tests/test_pipeline.py::test_memory_downcasting_preserves_values PASSED [ 60%]
tests/test_pipeline.py::test_uid_creation PASSED                        [ 70%]
tests/test_pipeline.py::test_frequency_encoding PASSED                  [ 80%]
tests/test_pipeline.py::test_full_pipeline_integration PASSED           [ 90%]

========================= 9 passed in 12.34s =========================
```

### Test Coverage

#### Test Case 1: `uid_TransactionFreq_24h` Validation

**What it tests:**
- Correct counting of transactions within 86,400-second window
- Cumulative counts for repeated transactions
- Exclusion of transactions outside 24h window

**Golden Dataset Pattern:**
```python
# Rows 10-15: Same uid, within 6000 seconds
# Expected: Counts should be [1, 2, 3, 4, 5, 6]

df.loc[10:15, 'card1'] = 1001
df.loc[10:15, 'addr1'] = 100
df.loc[10:15, 'D1'] = 5
df.loc[10:15, 'TransactionDT'] = base_time + 10000 + np.arange(6) * 1000
```

**Validation:**
```python
assert fraud_counts[0] >= 1  # First transaction sees itself
assert fraud_counts[5] >= 6  # Last transaction sees all 6
```

#### Test Case 2: `Amt_to_Mean_Ratio` NaN/Inf Handling

**What it tests:**
- Division by zero → 1.0 (neutral)
- NaN values → 1.0 (neutral)
- Inf values → 1.0 (neutral)

**Edge Cases:**
```python
# Case 1: No historical data (first transaction)
# mean = current amount, ratio = 1.0

# Case 2: Mean = 0
df.loc[0, 'uid_TransactionAmt_mean_30d'] = 0.0
# Expected: ratio = 1.0 (not Inf)

# Case 3: Amount = NaN
df.loc[0, 'TransactionAmt'] = np.nan
# Expected: ratio = 1.0
```

**Validation:**
```python
assert df['Amt_to_Mean_Ratio'].isna().sum() == 0  # No NaN
assert np.isinf(df['Amt_to_Mean_Ratio']).sum() == 0  # No Inf
```

#### Test Case 3: Memory Downcasting

**What it tests:**
- float64 → float32 conversion
- int64 → int8/int16/int32 optimization
- Value preservation (no data loss)

**Validation:**
```python
# Before
assert df['col_float64'].dtype == np.float64

# After optimization
optimized_df = reduce_mem_usage(df)
assert optimized_df['col_float64'].dtype == np.float32

# Values preserved (within float32 precision)
max_error = np.max(np.abs((original - optimized) / original))
assert max_error < 1e-6
```

### Running Tests in CI/CD

```yaml
# .github/workflows/test.yml
name: Unit Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - name: Set up Python
        uses: actions/setup-python@v2
        with:
          python-version: 3.8
      - name: Install dependencies
        run: |
          pip install -r requirements.txt
          pip install pytest pytest-cov
      - name: Run tests
        run: pytest tests/ -v --cov=. --cov-report=xml
      - name: Upload coverage
        uses: codecov/codecov-action@v2
```

### Manual Test Execution

```bash
# Run without pytest (standalone)
python tests/test_pipeline.py
```

Output:
```
SENTINEL UNIT TESTING SUITE
================================================================================

Creating golden dataset...
✓ Golden dataset created: (100, 8)

Running Test Case 1: uid_TransactionFreq_24h...
✓ Test Case 1 PASSED: uid_TransactionFreq_24h validated
✓ Test Case 1 (Edge Cases) PASSED: Edge cases validated

Running Test Case 2: Amt_to_Mean_Ratio...
✓ Test Case 2 PASSED: Amt_to_Mean_Ratio NaN/Inf handling validated
✓ Test Case 2 (Calculation) PASSED: Ratio calculation validated

Running Test Case 3: Memory Downcasting...
✓ Test Case 3 PASSED: Memory downcasting validated
✓ Test Case 3 (Value Preservation) PASSED: Integer values preserved

Running Additional Tests...
✓ Additional Test PASSED: UID creation validated
✓ Additional Test PASSED: Frequency encoding validated

Running Integration Test...
✓ Integration Test PASSED: Full pipeline validated

================================================================================
ALL TESTS PASSED ✓
================================================================================
```

---

## 📊 Production Monitoring

### Purpose

Detect feature drift and performance degradation in production using PSI and KL Divergence.

### Quick Start

```bash
python monitoring.py
```

### Expected Output

```
SENTINEL PRODUCTION MONITORING - DEMO
================================================================================

SCENARIO 1: Normal Production Data (No Drift)
================================================================================

FEATURE DRIFT DETECTION
================================================================================

Monitoring 5 features...
Production batch size: 1,000 transactions

Feature                        PSI        KL Div     Drift?
--------------------------------------------------------------------------------
TransactionAmt                 0.0234     0.0156     ✓ OK
uid_TransactionFreq_24h        0.0189     0.0123     ✓ OK
uid_TransactionAmt_mean_30d    0.0267     0.0178     ✓ OK
Amt_to_Mean_Ratio              0.0145     0.0098     ✓ OK
card1_freq                     0.0198     0.0134     ✓ OK
--------------------------------------------------------------------------------

✓ No significant drift detected

SCENARIO 2: Production Data with Moderate Drift
================================================================================

Feature                        PSI        KL Div     Drift?
--------------------------------------------------------------------------------
TransactionAmt                 0.2456     0.1678     ⚠ DRIFT
uid_TransactionFreq_24h        0.0189     0.0123     ✓ OK
uid_TransactionAmt_mean_30d    0.1234     0.0856     ✓ OK
Amt_to_Mean_Ratio              0.0987     0.0654     ✓ OK
card1_freq                     0.0198     0.0134     ✓ OK
--------------------------------------------------------------------------------

⚠ WARNING: Feature drift detected!
Recommended action: Investigate and consider model retraining
```

### Interpreting Drift Metrics

#### Population Stability Index (PSI)

| PSI Value | Interpretation | Action |
|-----------|----------------|--------|
| < 0.1 | No significant change | ✓ Continue monitoring |
| 0.1 - 0.2 | Moderate change | ⚠ Investigate |
| > 0.2 | Significant change | 🚨 Retrain model |

**Formula:**
```
PSI = Σ (% production - % reference) × ln(% production / % reference)
```

**Example:**
```
TransactionAmt Distribution:

Bin         Reference    Production    Contribution to PSI
────────────────────────────────────────────────────────────
$0-$50      30%          25%           0.0145
$50-$100    25%          30%           0.0189
$100-$200   20%          20%           0.0000
$200-$500   15%          15%           0.0000
$500+       10%          10%           0.0000
────────────────────────────────────────────────────────────
Total PSI:  0.0334 (No significant drift)
```

#### KL Divergence

| KL Value | Interpretation | Action |
|----------|----------------|--------|
| < 0.05 | Minimal drift | ✓ OK |
| 0.05 - 0.1 | Moderate drift | ⚠ Monitor closely |
| > 0.1 | Significant drift | 🚨 Investigate |

**Formula:**
```
KL(P || Q) = Σ P(x) × log(P(x) / Q(x))
```

Where:
- P = Production distribution
- Q = Reference (training) distribution

### Monitoring Report Structure

```json
{
  "report_timestamp": "2026-01-31T04:30:00",
  "model_version": "1.0.0",
  "production_batch_size": 1000,
  "drift_detection": {
    "timestamp": "2026-01-31T04:30:00",
    "drift_detected": true,
    "features": {
      "TransactionAmt": {
        "psi": 0.2456,
        "kl_divergence": 0.1678,
        "drift_detected": true,
        "psi_threshold": 0.2,
        "kl_threshold": 0.1
      }
    }
  },
  "performance_evaluation": {
    "roc_auc": 0.8934,
    "precision": 0.8123,
    "recall": 0.6789,
    "baseline_roc_auc": 0.9210,
    "roc_auc_delta": -0.0276,
    "performance_degraded": false
  },
  "alerts": [
    {
      "type": "FEATURE_DRIFT",
      "severity": "WARNING",
      "message": "Feature drift detected in production data",
      "action": "Investigate and consider model retraining"
    }
  ]
}
```

### Programmatic Usage

```python
from monitoring import SentinelMonitor

# Initialize monitor
monitor = SentinelMonitor(
    model_path="./models/sentinel_fraud_model.pkl",
    reference_data_path="./data/train_engineered.pkl",
    metadata_path="./models/model_metadata.json"
)

# Load production batch
production_df = pd.read_csv("production_batch.csv")

# Detect drift
drift_results = monitor.detect_feature_drift(
    production_df,
    features_to_monitor=['TransactionAmt', 'uid_TransactionFreq_24h'],
    psi_threshold=0.2,
    kl_threshold=0.1
)

# Evaluate performance (if labels available)
performance = monitor.evaluate_performance(production_df, threshold=0.5)

# Generate full report
report = monitor.generate_monitoring_report(
    production_df,
    output_path="./logs/monitoring_report.json"
)

# Check for alerts
if report['alerts']:
    for alert in report['alerts']:
        print(f"[{alert['severity']}] {alert['message']}")
        # Send to alerting system (PagerDuty, Slack, etc.)
```

### Integration with Alerting Systems

#### Slack Webhook

```python
import requests

def send_slack_alert(report):
    if report['alerts']:
        webhook_url = "https://hooks.slack.com/services/YOUR/WEBHOOK/URL"
        
        message = {
            "text": f"🚨 Sentinel Model Alert - {len(report['alerts'])} issues detected",
            "attachments": [
                {
                    "color": "danger" if alert['severity'] == "CRITICAL" else "warning",
                    "fields": [
                        {"title": "Type", "value": alert['type'], "short": True},
                        {"title": "Severity", "value": alert['severity'], "short": True},
                        {"title": "Message", "value": alert['message']},
                        {"title": "Action", "value": alert['action']}
                    ]
                }
                for alert in report['alerts']
            ]
        }
        
        requests.post(webhook_url, json=message)
```

#### PagerDuty

```python
from pdpyras import EventsAPISession

def send_pagerduty_alert(report):
    session = EventsAPISession('YOUR_INTEGRATION_KEY')
    
    for alert in report['alerts']:
        if alert['severity'] == 'CRITICAL':
            session.trigger(
                summary=alert['message'],
                source='Sentinel Monitoring',
                severity='critical',
                custom_details=alert
            )
```

### Scheduled Monitoring (Cron)

```bash
# crontab -e

# Run monitoring every hour
0 * * * * cd /path/to/sentinel && python monitoring.py >> /var/log/sentinel_monitoring.log 2>&1

# Run monitoring every 6 hours
0 */6 * * * cd /path/to/sentinel && python monitoring.py
```

---

## 🔄 Complete MLOps Workflow

### 1. Development Phase

```bash
# Run unit tests
pytest tests/test_pipeline.py -v

# Train model
python model_training.py

# Generate SHAP explanations
python interpretability.py
```

### 2. Deployment Phase

```bash
# Evaluate model
python model_evaluation.py

# Initial monitoring baseline
python monitoring.py
```

### 3. Production Phase

```bash
# Hourly monitoring
python monitoring.py

# Weekly SHAP analysis
python interpretability.py

# Continuous testing (CI/CD)
pytest tests/ -v
```

---

## 📁 Generated Artifacts

```
sentinel/
├── reports/
│   ├── shap_summary.png              # Global feature importance
│   ├── fraud_explanation.png         # Fraud transaction explanation
│   ├── legitimate_explanation.png    # Legitimate transaction explanation
│   └── explanations/                 # Batch explanations
│       ├── transaction_X_explanation.png
│       └── batch_explanations.json
├── logs/
│   ├── monitoring_report_normal.json
│   ├── monitoring_report_drift.json
│   └── monitoring_report_severe.json
└── tests/
    └── test_pipeline.py
```

---

## 🐛 Troubleshooting

### Issue: SHAP computation too slow

**Solution:**
- Reduce `background_size` from 100 to 50
- Reduce `sample_size` in summary plot from 1000 to 500
- Use `tree_path_dependent` feature perturbation (default)

### Issue: Tests failing due to floating point precision

**Solution:**
```python
# Use approximate equality
assert abs(expected - actual) < 1e-6

# Or use numpy.allclose
assert np.allclose(expected, actual, rtol=1e-5)
```

### Issue: Monitoring reports show false drift

**Solution:**
- Increase PSI threshold from 0.2 to 0.25
- Use larger production batches (>1000 samples)
- Check for data quality issues in production

---

## 📚 Additional Resources

- **SHAP Documentation**: https://shap.readthedocs.io/
- **Pytest Documentation**: https://docs.pytest.org/
- **PSI Calculation**: https://mwburke.github.io/data%20science/2018/04/29/population-stability-index.html
- **Model Monitoring Best Practices**: https://christophergs.com/machine%20learning/2020/03/14/how-to-monitor-machine-learning-models/

---

**Last Updated**: 2026-01-31  
**Version**: 2.0.0

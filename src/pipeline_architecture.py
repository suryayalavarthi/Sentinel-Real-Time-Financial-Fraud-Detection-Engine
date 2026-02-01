"""
Sentinel Pipeline Architecture Visualization

This module provides a visual representation of the data flow
through the Sentinel fraud detection pipeline.
"""


def print_pipeline_architecture():
    """Print ASCII diagram of the Sentinel pipeline."""
    
    diagram = """
╔══════════════════════════════════════════════════════════════════════════════╗
║                    SENTINEL FRAUD DETECTION PIPELINE                         ║
╚══════════════════════════════════════════════════════════════════════════════╝

┌─────────────────────────────────────────────────────────────────────────────┐
│ STAGE 1: DATA INGESTION (data_ingestion.py)                                │
└─────────────────────────────────────────────────────────────────────────────┘

    ┌──────────────────────┐         ┌──────────────────────┐
    │ train_transaction.csv│         │  train_identity.csv  │
    │   (590K rows)        │         │   (144K rows)        │
    │   float64/int64      │         │   float64/int64      │
    └──────────┬───────────┘         └──────────┬───────────┘
               │                                │
               ▼                                ▼
    ┌──────────────────────┐         ┌──────────────────────┐
    │  reduce_mem_usage()  │         │  reduce_mem_usage()  │
    │  ↓ 65% memory        │         │  ↓ 65% memory        │
    └──────────┬───────────┘         └──────────┬───────────┘
               │                                │
               └────────────┬───────────────────┘
                            ▼
                ┌───────────────────────┐
                │   LEFT JOIN           │
                │   on TransactionID    │
                │   (preserve all txns) │
                └───────────┬───────────┘
                            ▼
                ┌───────────────────────┐
                │ CHRONOLOGICAL SORT    │
                │ by TransactionDT      │
                │ (prevent leakage)     │
                └───────────┬───────────┘
                            ▼
                ┌───────────────────────┐
                │  train_optimized.pkl  │
                │  (590K × 434 cols)    │
                │  Memory: ~280 MB      │
                └───────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│ STAGE 2: FEATURE ENGINEERING (feature_engineering.py)                      │
└─────────────────────────────────────────────────────────────────────────────┘

    ┌───────────────────────┐
    │  train_optimized.pkl  │
    └───────────┬───────────┘
                ▼
    ┌───────────────────────────────────────────────────────┐
    │ STEP 1: Create UID                                    │
    │ uid = card1 + addr1 + D1                              │
    │ Purpose: Pseudo-user identifier for velocity tracking│
    └───────────┬───────────────────────────────────────────┘
                ▼
    ┌───────────────────────────────────────────────────────┐
    │ STEP 2: Velocity Features (Time-Series)              │
    │                                                       │
    │ ┌─────────────────────────────────────────────────┐  │
    │ │ uid_TransactionFreq_24h                         │  │
    │ │ • Count transactions in last 86,400 seconds     │  │
    │ │ • Per uid group                                 │  │
    │ │ • Type: int16                                   │  │
    │ └─────────────────────────────────────────────────┘  │
    │                                                       │
    │ ┌─────────────────────────────────────────────────┐  │
    │ │ uid_TransactionAmt_mean_30d                     │  │
    │ │ • Rolling mean over 2,592,000 seconds           │  │
    │ │ • Per uid group                                 │  │
    │ │ • Type: float32                                 │  │
    │ └─────────────────────────────────────────────────┘  │
    └───────────┬───────────────────────────────────────────┘
                ▼
    ┌───────────────────────────────────────────────────────┐
    │ STEP 3: Divergence Features                          │
    │                                                       │
    │ ┌─────────────────────────────────────────────────┐  │
    │ │ Amt_to_Mean_Ratio                               │  │
    │ │ • TransactionAmt / uid_TransactionAmt_mean_30d  │  │
    │ │ • Fraud Signal: Ratio > 3x                      │  │
    │ │ • NaN handling: Replace with 1.0                │  │
    │ └─────────────────────────────────────────────────┘  │
    └───────────┬───────────────────────────────────────────┘
                ▼
    ┌───────────────────────────────────────────────────────┐
    │ STEP 4: Frequency Encoding                           │
    │                                                       │
    │ ┌─────────────────────────────────────────────────┐  │
    │ │ card1_freq                                      │  │
    │ │ • Count of card1 occurrences                    │  │
    │ │ • Rare cards = higher fraud risk                │  │
    │ └─────────────────────────────────────────────────┘  │
    │                                                       │
    │ ┌─────────────────────────────────────────────────┐  │
    │ │ P_emaildomain_freq                              │  │
    │ │ • Count of email domain occurrences             │  │
    │ │ • Rare domains = higher fraud risk              │  │
    │ └─────────────────────────────────────────────────┘  │
    └───────────┬───────────────────────────────────────────┘
                ▼
    ┌───────────────────────────────────────────────────────┐
    │ STEP 5: Final Memory Optimization                    │
    │ • Downcast all numeric types                         │
    │ • gc.collect()                                       │
    └───────────┬───────────────────────────────────────────┘
                ▼
    ┌───────────────────────┐
    │ train_engineered.pkl  │
    │ (590K × 441 cols)     │
    │ Memory: ~320 MB       │
    └───────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│ FEATURE SUMMARY                                                             │
└─────────────────────────────────────────────────────────────────────────────┘

    Original Features:     434
    Engineered Features:   +7
    ─────────────────────────
    Total Features:        441

    New Features:
    ✓ uid                           (Unique identifier)
    ✓ uid_TransactionFreq_24h       (Velocity: 24h count)
    ✓ uid_TransactionAmt_mean_30d   (Velocity: 30d mean)
    ✓ Amt_to_Mean_Ratio             (Divergence: amount ratio)
    ✓ card1_freq                    (Encoding: card frequency)
    ✓ P_emaildomain_freq            (Encoding: domain frequency)

┌─────────────────────────────────────────────────────────────────────────────┐
│ MEMORY OPTIMIZATION SUMMARY                                                 │
└─────────────────────────────────────────────────────────────────────────────┘

    Stage                      Memory      Reduction
    ─────────────────────────────────────────────────
    Raw CSV (float64)          800 MB      -
    After Ingestion            280 MB      65%
    After Feature Eng          320 MB      60% (vs raw)
    
    Peak Memory Usage:         ~900 MB (during rolling windows)
    Safe for:                  2GB+ RAM systems

┌─────────────────────────────────────────────────────────────────────────────┐
│ TIME COMPLEXITY ANALYSIS                                                    │
└─────────────────────────────────────────────────────────────────────────────┘

    Operation                  Complexity       Notes
    ──────────────────────────────────────────────────────────────
    Data Loading               O(n)             CSV parsing
    Memory Optimization        O(n × m)         n=rows, m=cols
    UID Creation               O(n)             String concat
    24h Frequency              O(n × k)         k=avg txns per uid
    30d Rolling Mean           O(n × k)         Same as above
    Frequency Encoding         O(n)             Hash map lookup
    ──────────────────────────────────────────────────────────────
    Total Pipeline             O(n × k)         k typically < 100

    Expected Runtime (590K rows, 8GB RAM):
    • Data Ingestion:      ~45 seconds
    • Feature Engineering: ~120 seconds
    • Total:               ~3 minutes

╔══════════════════════════════════════════════════════════════════════════════╗
║                           NEXT STEPS                                         ║
╚══════════════════════════════════════════════════════════════════════════════╝

    1. Load processed data:
       df = pd.read_pickle(\1../data/processed/train_engineered.pkl')
    
    2. Exploratory Data Analysis:
       • Analyze feature distributions
       • Check for multicollinearity
       • Identify top fraud signals
    
    3. Model Training:
       • XGBoost / LightGBM
       • Time-series cross-validation (5-fold)
       • Hyperparameter tuning
    
    4. Evaluation:
       • ROC-AUC score
       • Precision-Recall curve
       • Feature importance (SHAP)

"""
    
    print(diagram)


if __name__ == "__main__":
    print_pipeline_architecture()

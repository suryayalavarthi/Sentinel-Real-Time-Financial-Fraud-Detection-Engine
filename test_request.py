import requests
import pandas as pd
import numpy as np
import json

URL = "http://localhost:8080/predict"
data_path = "/Users/suryayalavarthi/Downloads/sentinel/ieee-fraud-detection/processed/train_engineered.pkl"

# 1. Load data
df = pd.read_pickle(data_path)

# 2. Get a sample row
sample_df = df.drop(columns=['isFraud']) if 'isFraud' in df.columns else df

# DROP THE METADATA COLUMNS THAT THE API REJECTS
cols_to_drop = ['TransactionID', 'TransactionDT', 'uid']
sample_df = sample_df.drop(columns=[c for c in cols_to_drop if c in sample_df.columns])

row = sample_df.iloc[[0]].copy()

# 3. ENCODE STRINGS TO NUMBERS (Crucial Step)
# This mimics your training preprocessing
for col in row.select_dtypes(include=['object', 'category']).columns:
    row[col] = row[col].astype('category').cat.codes

# 4. Clean and Convert to Dict
sample_row = row.iloc[0].to_dict()
clean_sample = {k: float(v) if not pd.isna(v) else 0.0 for k, v in sample_row.items()}

payload = {"features": clean_sample}

print(f"🚀 Sending real encoded transaction to Sentinel...")
response = requests.post(URL, json=payload)

if response.status_code == 200:
    print("✅ SUCCESS: Prediction Received!")
    print(json.dumps(response.json(), indent=2))
else:
    print(f"❌ Failed with status {response.status_code}")
    print(response.text)
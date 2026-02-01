"""
Sentinel - Model Interpretability with SHAP
Explain fraud detection decisions using SHapley Additive exPlanations
Purpose: Provide transparent, actionable explanations for model predictions
Constraints: Memory-efficient SHAP computation using background sampling
"""

import json
import warnings
from typing import Dict, List, Optional, Tuple

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

warnings.filterwarnings('ignore')


class SentinelExplainer:
    """
    SHAP-based explainer for Sentinel fraud detection model.
    
    Engineering Design:
    - Uses TreeExplainer (optimized for XGBoost)
    - Background dataset sampling (100-500 rows) for memory efficiency
    - Caches explainer to avoid recomputation
    """

    @staticmethod
    def _encode_for_shap(df: pd.DataFrame) -> pd.DataFrame:
        """
        Convert categorical/object columns into numeric codes for SHAP.
        SHAP expects numeric arrays and will error on object dtypes.
        """
        encoded = df.copy()
        for col in encoded.columns:
            if pd.api.types.is_categorical_dtype(encoded[col]):
                encoded[col] = encoded[col].cat.codes
            elif encoded[col].dtype == object:
                encoded[col] = encoded[col].astype('category').cat.codes
        return encoded
    
    def __init__(
        self,
        model_path: str = "./models/sentinel_fraud_model.pkl",
        data_path: str = "./data/train_engineered.pkl",
        background_size: int = 100
    ):
        """
        Initialize SHAP explainer.
        
        Args:
            model_path: Path to trained model
            data_path: Path to training data
            background_size: Number of samples for SHAP background dataset
        """
        print("\n" + "=" * 80)
        print("SENTINEL MODEL INTERPRETABILITY - SHAP INITIALIZATION")
        print("=" * 80)
        
        # Load model
        print(f"\n[1/4] Loading model from {model_path}...")
        self.model = joblib.load(model_path)
        print("✓ Model loaded")
        
        # Load data
        print(f"\n[2/4] Loading data from {data_path}...")
        df = pd.read_pickle(data_path)
        print(f"✓ Data loaded: {df.shape}")
        
        # Prepare features
        print("\n[3/4] Preparing features...")
        from model_training import prepare_features_and_target
        self.X, self.y = prepare_features_and_target(df)
        
        # Load feature names
        try:
            with open('./models/feature_names.json') as f:
                self.feature_names = json.load(f)
        except:
            self.feature_names = self.X.columns.tolist()
        
        print(f"✓ Features prepared: {len(self.feature_names)} features")
        
        # Create background dataset (stratified sampling)
        print(f"\n[4/4] Creating SHAP background dataset ({background_size} samples)...")
        
        # Stratified sampling: ensure both fraud and legitimate transactions
        fraud_indices = self.y[self.y == 1].index
        legit_indices = self.y[self.y == 0].index
        
        # Sample proportionally
        n_fraud = min(background_size // 10, len(fraud_indices))  # ~10% fraud
        n_legit = background_size - n_fraud
        
        fraud_sample = np.random.choice(fraud_indices, n_fraud, replace=False)
        legit_sample = np.random.choice(legit_indices, n_legit, replace=False)
        
        background_indices = np.concatenate([fraud_sample, legit_sample])
        self.background_data = self._encode_for_shap(self.X.iloc[background_indices])
        
        print(f"✓ Background dataset created:")
        print(f"  - Fraud samples: {n_fraud}")
        print(f"  - Legitimate samples: {n_legit}")
        
        # Initialize SHAP explainer
        print("\nInitializing SHAP TreeExplainer...")
        self.explainer = shap.TreeExplainer(
            self.model,
            feature_perturbation='tree_path_dependent'
        )
        print("✓ SHAP explainer initialized")
        
        print("\n" + "=" * 80)
        print("INITIALIZATION COMPLETE")
        print("=" * 80 + "\n")
    
    def generate_summary_plot(
        self,
        sample_size: int = 1000,
        max_display: int = 20,
        save_path: str = "./reports/shap_summary.png"
    ) -> None:
        """
        Generate SHAP summary plot showing feature importance.
        
        Engineering Rationale:
        - Summary plot shows global feature importance
        - Each dot represents a transaction
        - Color indicates feature value (red=high, blue=low)
        - Position indicates SHAP value (impact on prediction)
        
        Args:
            sample_size: Number of samples to explain
            max_display: Maximum features to display
            save_path: Path to save plot
        """
        print("\n" + "=" * 80)
        print("GENERATING SHAP SUMMARY PLOT")
        print("=" * 80)
        
        # Sample data for explanation
        print(f"\n[1/3] Sampling {sample_size} transactions...")
        sample_indices = np.random.choice(len(self.X), sample_size, replace=False)
        X_sample = self.X.iloc[sample_indices]
        
        # Calculate SHAP values
        print("\n[2/3] Computing SHAP values (this may take a few minutes)...")
        X_sample_encoded = self._encode_for_shap(X_sample)
        shap_values = self.explainer.shap_values(X_sample_encoded)
        
        # Generate plot
        print(f"\n[3/3] Generating summary plot (top {max_display} features)...")
        
        plt.figure(figsize=(12, 8))
        shap.summary_plot(
            shap_values,
            X_sample_encoded,
            feature_names=self.feature_names,
            max_display=max_display,
            show=False
        )
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved: {save_path}")
        
        plt.close()
        
        # Print top features
        print("\n" + "-" * 80)
        print("TOP FEATURES BY MEAN |SHAP VALUE|")
        print("-" * 80)
        
        mean_abs_shap = np.abs(shap_values).mean(axis=0)
        feature_importance = pd.DataFrame({
            'feature': self.feature_names,
            'mean_abs_shap': mean_abs_shap
        }).sort_values('mean_abs_shap', ascending=False)
        
        print(feature_importance.head(10).to_string(index=False))
        
        print("\n" + "=" * 80)
        print("SUMMARY PLOT COMPLETE")
        print("=" * 80 + "\n")
    
    def explain_transaction(
        self,
        transaction_idx: int,
        plot_type: str = 'waterfall',
        save_path: Optional[str] = None
    ) -> Dict:
        """
        Explain a single transaction prediction.
        
        Engineering Rationale:
        - Waterfall plot shows cumulative impact of features
        - Force plot shows how features push prediction from base value
        - Provides actionable insights for fraud analysts
        
        Args:
            transaction_idx: Index of transaction to explain
            plot_type: 'waterfall' or 'force'
            save_path: Optional path to save plot
        
        Returns:
            Dictionary with explanation details
        """
        print("\n" + "=" * 80)
        print(f"EXPLAINING TRANSACTION {transaction_idx}")
        print("=" * 80)
        
        # Get transaction data
        transaction = self.X.iloc[transaction_idx:transaction_idx+1]
        true_label = self.y.iloc[transaction_idx]
        
        # Get prediction
        pred_proba = self.model.predict_proba(transaction)[0][1]
        pred_label = int(pred_proba >= 0.5)
        
        print(f"\nTransaction Details:")
        print(f"  True Label: {'FRAUD' if true_label == 1 else 'LEGITIMATE'}")
        print(f"  Predicted: {'FRAUD' if pred_label == 1 else 'LEGITIMATE'}")
        print(f"  Fraud Probability: {pred_proba:.4f}")
        
        # Calculate SHAP values
        print("\nComputing SHAP values...")
        transaction_encoded = self._encode_for_shap(transaction)
        shap_values = self.explainer.shap_values(transaction_encoded)
        
        # Get base value (expected value)
        base_value = self.explainer.expected_value
        
        # Create explanation object
        explanation = shap.Explanation(
            values=shap_values[0],
            base_values=base_value,
            data=transaction.values[0],
            feature_names=self.feature_names
        )
        
        # Generate plot
        print(f"\nGenerating {plot_type} plot...")
        
        if plot_type == 'waterfall':
            plt.figure(figsize=(10, 8))
            shap.waterfall_plot(explanation, max_display=15, show=False)
            
        elif plot_type == 'force':
            # Force plot requires matplotlib=True for static image
            shap.force_plot(
                base_value,
                shap_values[0],
                transaction.iloc[0],
                feature_names=self.feature_names,
                matplotlib=True,
                show=False
            )
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Saved: {save_path}")
        else:
            plt.show()
        
        plt.close()
        
        # Get top contributing features
        feature_contributions = pd.DataFrame({
            'feature': self.feature_names,
            'shap_value': shap_values[0],
            'feature_value': transaction.values[0]
        }).sort_values('shap_value', key=abs, ascending=False)
        
        print("\n" + "-" * 80)
        print("TOP CONTRIBUTING FEATURES")
        print("-" * 80)
        print(feature_contributions.head(10).to_string(index=False))
        
        # Create explanation dictionary
        explanation_dict = {
            'transaction_idx': transaction_idx,
            'true_label': int(true_label),
            'predicted_label': pred_label,
            'fraud_probability': float(pred_proba),
            'base_value': float(base_value),
            'top_features': feature_contributions.head(10).to_dict('records')
        }
        
        print("\n" + "=" * 80)
        print("EXPLANATION COMPLETE")
        print("=" * 80 + "\n")
        
        return explanation_dict
    
    def explain_transaction_by_id(
        self,
        transaction_id: int,
        plot_type: str = 'waterfall',
        save_path: Optional[str] = None
    ) -> Dict:
        """
        Explain transaction by TransactionID (if available in data).
        
        Args:
            transaction_id: TransactionID value
            plot_type: 'waterfall' or 'force'
            save_path: Optional path to save plot
        
        Returns:
            Dictionary with explanation details
        """
        # Load original data to get TransactionID
        df = pd.read_pickle("./data/train_engineered.pkl")
        
        if 'TransactionID' not in df.columns:
            raise ValueError("TransactionID column not found in data")
        
        # Find index
        idx = df[df['TransactionID'] == transaction_id].index
        
        if len(idx) == 0:
            raise ValueError(f"TransactionID {transaction_id} not found")
        
        transaction_idx = idx[0]
        
        return self.explain_transaction(transaction_idx, plot_type, save_path)
    
    def batch_explain(
        self,
        n_samples: int = 10,
        output_dir: str = "./reports/explanations"
    ) -> List[Dict]:
        """
        Generate explanations for multiple transactions.
        
        Args:
            n_samples: Number of transactions to explain
            output_dir: Directory to save explanations
        
        Returns:
            List of explanation dictionaries
        """
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        print("\n" + "=" * 80)
        print(f"BATCH EXPLANATION - {n_samples} TRANSACTIONS")
        print("=" * 80)
        
        # Sample fraud and legitimate transactions
        fraud_indices = self.y[self.y == 1].index[:n_samples//2]
        legit_indices = self.y[self.y == 0].index[:n_samples//2]
        
        sample_indices = np.concatenate([fraud_indices, legit_indices])
        
        explanations = []
        
        for i, idx in enumerate(sample_indices, 1):
            print(f"\n[{i}/{len(sample_indices)}] Explaining transaction {idx}...")
            
            save_path = f"{output_dir}/transaction_{idx}_explanation.png"
            explanation = self.explain_transaction(idx, save_path=save_path)
            explanations.append(explanation)
        
        # Save explanations as JSON
        json_path = f"{output_dir}/batch_explanations.json"
        with open(json_path, 'w') as f:
            json.dump(explanations, f, indent=2)
        
        print(f"\n✓ Saved batch explanations: {json_path}")
        
        print("\n" + "=" * 80)
        print("BATCH EXPLANATION COMPLETE")
        print("=" * 80 + "\n")
        
        return explanations


def main():
    """Execute SHAP interpretability analysis."""
    
    # Check if SHAP is installed
    try:
        import shap
    except ImportError:
        print("\n⚠ ERROR: SHAP library not installed")
        print("\nInstall with:")
        print("  pip install shap")
        return
    
    # Initialize explainer
    explainer = SentinelExplainer(
        model_path="./models/sentinel_fraud_model.pkl",
        data_path="./data/train_engineered.pkl",
        background_size=100
    )
    
    # Generate summary plot
    explainer.generate_summary_plot(
        sample_size=1000,
        max_display=20,
        save_path="./reports/shap_summary.png"
    )
    
    # Explain individual transactions
    print("\n" + "=" * 80)
    print("INDIVIDUAL TRANSACTION EXPLANATIONS")
    print("=" * 80)
    
    # Explain a fraud transaction
    fraud_idx = explainer.y[explainer.y == 1].index[0]
    explainer.explain_transaction(
        fraud_idx,
        plot_type='waterfall',
        save_path="./reports/fraud_explanation.png"
    )
    
    # Explain a legitimate transaction
    legit_idx = explainer.y[explainer.y == 0].index[0]
    explainer.explain_transaction(
        legit_idx,
        plot_type='waterfall',
        save_path="./reports/legitimate_explanation.png"
    )
    
    print("\n" + "=" * 80)
    print("INTERPRETABILITY ANALYSIS COMPLETE")
    print("=" * 80)
    print("\n📊 Generated Reports:")
    print("  - ./reports/shap_summary.png")
    print("  - ./reports/fraud_explanation.png")
    print("  - ./reports/legitimate_explanation.png")
    print("\n" + "=" * 80 + "\n")


if __name__ == "__main__":
    main()

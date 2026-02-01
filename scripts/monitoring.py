"""
Sentinel - Production Monitoring & Drift Detection
Monitor model performance and detect feature/concept drift
Purpose: Ensure model reliability in production through continuous monitoring
Metrics: PSI, KL Divergence, performance degradation detection
"""

import json
import warnings
from pathlib import Path
import sys
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from scipy import stats
from sklearn.metrics import roc_auc_score, precision_score, recall_score

warnings.filterwarnings('ignore')

SRC_PATH = Path(__file__).resolve().parents[1] / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))


class SentinelMonitor:
    """
    Production monitoring system for Sentinel fraud detection model.
    
    Engineering Design:
    - Tracks feature drift using PSI and KL Divergence
    - Monitors model performance degradation
    - Generates JSON logs for alerting systems
    - Supports batch and real-time monitoring
    """
    
    def __init__(
        self,
        model_path: str = "./models/sentinel_fraud_model.pkl",
        reference_data_path: str = "./data/processed/train_engineered.pkl",
        metadata_path: str = "./models/model_metadata.json"
    ):
        """
        Initialize monitoring system.
        
        Args:
            model_path: Path to trained model
            reference_data_path: Path to training/reference data
            metadata_path: Path to model metadata
        """
        print("\n" + "=" * 80)
        print("SENTINEL PRODUCTION MONITORING - INITIALIZATION")
        print("=" * 80)
        
        # Load model
        print(f"\n[1/3] Loading model from {model_path}...")
        self.model = joblib.load(model_path)
        print("✓ Model loaded")
        
        # Load reference data
        print(f"\n[2/3] Loading reference data from {reference_data_path}...")
        self.reference_df = pd.read_pickle(reference_data_path)
        print(f"✓ Reference data loaded: {self.reference_df.shape}")
        
        # Load metadata
        print(f"\n[3/3] Loading metadata from {metadata_path}...")
        with open(metadata_path) as f:
            self.metadata = json.load(f)
        
        print(f"✓ Metadata loaded:")
        print(f"  Model Version: {self.metadata.get('model_version', 'N/A')}")
        print(f"  Training Date: {self.metadata.get('training_date', 'N/A')}")
        print(f"  Baseline ROC-AUC: {self.metadata.get('avg_roc_auc', 'N/A')}")
        
        # Prepare reference features
        from model_training import prepare_features_and_target
        self.X_ref, self.y_ref = prepare_features_and_target(self.reference_df)
        
        print("\n" + "=" * 80)
        print("INITIALIZATION COMPLETE")
        print("=" * 80 + "\n")
    
    @staticmethod
    def _encode_for_model(df: pd.DataFrame) -> pd.DataFrame:
        encoded = df.copy()
        for col in encoded.columns:
            if pd.api.types.is_categorical_dtype(encoded[col]):
                encoded[col] = encoded[col].cat.codes
            elif encoded[col].dtype == object:
                encoded[col] = encoded[col].astype("category").cat.codes
        return encoded

    def calculate_psi(
        self,
        reference: np.ndarray,
        production: np.ndarray,
        bins: int = 10
    ) -> float:
        """
        Calculate Population Stability Index (PSI).
        
        PSI Formula:
        PSI = Σ (% production - % reference) × ln(% production / % reference)
        
        Interpretation:
        - PSI < 0.1: No significant change
        - 0.1 ≤ PSI < 0.2: Moderate change (investigate)
        - PSI ≥ 0.2: Significant change (retrain model)
        
        Args:
            reference: Reference distribution (training data)
            production: Production distribution (new data)
            bins: Number of bins for discretization
        
        Returns:
            PSI value
        """
        # Remove NaN values
        reference = reference[~np.isnan(reference)]
        production = production[~np.isnan(production)]
        
        # Create bins based on reference data
        _, bin_edges = np.histogram(reference, bins=bins)
        
        # Ensure bins cover both distributions
        min_val = min(reference.min(), production.min())
        max_val = max(reference.max(), production.max())
        bin_edges[0] = min_val - 1e-6
        bin_edges[-1] = max_val + 1e-6
        
        # Calculate distributions
        ref_counts, _ = np.histogram(reference, bins=bin_edges)
        prod_counts, _ = np.histogram(production, bins=bin_edges)
        
        # Convert to percentages
        ref_percents = ref_counts / len(reference)
        prod_percents = prod_counts / len(production)
        
        # Avoid division by zero
        ref_percents = np.where(ref_percents == 0, 0.0001, ref_percents)
        prod_percents = np.where(prod_percents == 0, 0.0001, prod_percents)
        
        # Calculate PSI
        psi = np.sum((prod_percents - ref_percents) * np.log(prod_percents / ref_percents))
        
        return float(psi)
    
    def calculate_kl_divergence(
        self,
        reference: np.ndarray,
        production: np.ndarray,
        bins: int = 10
    ) -> float:
        """
        Calculate Kullback-Leibler Divergence.
        
        KL Divergence measures how one probability distribution diverges from
        a reference distribution.
        
        Interpretation:
        - KL = 0: Identical distributions
        - KL > 0: Distributions differ (higher = more different)
        - KL > 0.1: Significant drift
        
        Args:
            reference: Reference distribution
            production: Production distribution
            bins: Number of bins
        
        Returns:
            KL Divergence value
        """
        # Remove NaN
        reference = reference[~np.isnan(reference)]
        production = production[~np.isnan(production)]
        
        # Create bins
        _, bin_edges = np.histogram(reference, bins=bins)
        
        min_val = min(reference.min(), production.min())
        max_val = max(reference.max(), production.max())
        bin_edges[0] = min_val - 1e-6
        bin_edges[-1] = max_val + 1e-6
        
        # Calculate distributions
        ref_counts, _ = np.histogram(reference, bins=bin_edges)
        prod_counts, _ = np.histogram(production, bins=bin_edges)
        
        # Convert to probabilities
        ref_probs = ref_counts / len(reference)
        prod_probs = prod_counts / len(production)
        
        # Avoid log(0)
        ref_probs = np.where(ref_probs == 0, 1e-10, ref_probs)
        prod_probs = np.where(prod_probs == 0, 1e-10, prod_probs)
        
        # Calculate KL Divergence
        kl_div = np.sum(prod_probs * np.log(prod_probs / ref_probs))
        
        return float(kl_div)
    
    def detect_feature_drift(
        self,
        production_df: pd.DataFrame,
        features_to_monitor: Optional[List[str]] = None,
        psi_threshold: float = 0.2,
        kl_threshold: float = 0.1,
        log_path: str = "./logs/drift_detection.json"
    ) -> Dict:
        """
        Detect feature drift in production data.
        
        Args:
            production_df: Production data batch
            features_to_monitor: List of features to monitor (None = all)
            psi_threshold: PSI threshold for drift detection
            kl_threshold: KL Divergence threshold
        
        Returns:
            Dictionary with drift metrics
        """
        print("\n" + "=" * 80)
        print("FEATURE DRIFT DETECTION")
        print("=" * 80)
        
        # Prepare production features
        from model_training import prepare_features_and_target
        X_prod, _ = prepare_features_and_target(production_df)
        X_prod = self._encode_for_model(X_prod)
        
        # Default: Monitor all numerical features
        if features_to_monitor is None:
            features_to_monitor = [
                'TransactionAmt',
                'uid_TransactionFreq_24h',
                'uid_TransactionAmt_mean_30d',
                'Amt_to_Mean_Ratio',
                'card1_freq'
            ]
        
        drift_results = {
            'timestamp': datetime.now().isoformat(),
            'n_production_samples': len(X_prod),
            'features_monitored': len(features_to_monitor),
            'drift_detected': False,
            'features': {}
        }
        
        print(f"\nMonitoring {len(features_to_monitor)} features...")
        print(f"Production batch size: {len(X_prod):,} transactions\n")
        
        print(f"{'Feature':<30} {'PSI':<10} {'KL Div':<10} {'Drift?':<10}")
        print("-" * 80)
        
        for feature in features_to_monitor:
            if feature not in self.X_ref.columns or feature not in X_prod.columns:
                continue
            
            # Get distributions
            ref_values = self.X_ref[feature].values
            prod_values = X_prod[feature].values
            
            # Calculate metrics
            psi = self.calculate_psi(ref_values, prod_values)
            kl_div = self.calculate_kl_divergence(ref_values, prod_values)
            
            # Detect drift
            drift = psi >= psi_threshold or kl_div >= kl_threshold
            
            if drift:
                drift_results['drift_detected'] = True
            
            # Store results
            drift_results['features'][feature] = {
                'psi': float(psi),
                'kl_divergence': float(kl_div),
                'drift_detected': bool(drift),
                'psi_threshold': psi_threshold,
                'kl_threshold': kl_threshold
            }
            
            # Print results
            drift_status = "⚠ DRIFT" if drift else "✓ OK"
            print(f"{feature:<30} {psi:<10.4f} {kl_div:<10.4f} {drift_status:<10}")
        
        print("-" * 80)
        
        if drift_results['drift_detected']:
            print("\n⚠ WARNING: Feature drift detected!")
            print("Recommended action: Investigate and consider model retraining")
        else:
            print("\n✓ No significant drift detected")
        
        print("\n" + "=" * 80)
        print("DRIFT DETECTION COMPLETE")
        print("=" * 80 + "\n")

        # Persist drift results for monitoring logs
        import os
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        with open(log_path, 'w') as f:
            json.dump(drift_results, f, indent=2)
        print(f"✓ Drift log saved: {log_path}")

        return drift_results
    
    def evaluate_performance(
        self,
        production_df: pd.DataFrame,
        threshold: float = 0.5
    ) -> Dict:
        """
        Evaluate model performance on production data.
        
        Args:
            production_df: Production data with ground truth labels
            threshold: Classification threshold
        
        Returns:
            Dictionary with performance metrics
        """
        print("\n" + "=" * 80)
        print("PERFORMANCE EVALUATION")
        print("=" * 80)
        
        # Prepare features
        from model_training import prepare_features_and_target
        X_prod, y_prod = prepare_features_and_target(production_df)
        X_prod = self._encode_for_model(X_prod)
        
        # Predictions
        print("\nGenerating predictions...")
        booster = self.model.get_booster()
        expected_features = booster.feature_names or list(X_prod.columns)
        X_prod = X_prod[expected_features]
        dmatrix = xgb.DMatrix(X_prod, feature_names=expected_features)
        y_pred_proba = booster.predict(dmatrix)
        y_pred = (y_pred_proba >= threshold).astype(int)
        
        # Calculate metrics
        print("Calculating metrics...")
        
        roc_auc = roc_auc_score(y_prod, y_pred_proba)
        precision = precision_score(y_prod, y_pred, zero_division=0)
        recall = recall_score(y_prod, y_pred, zero_division=0)
        
        # Compare to baseline
        baseline_roc_auc = self.metadata.get('avg_roc_auc', 0.92)
        roc_auc_delta = roc_auc - baseline_roc_auc
        
        # Detect performance degradation
        degradation_threshold = 0.05  # 5% drop
        performance_degraded = roc_auc_delta < -degradation_threshold
        
        results = {
            'timestamp': datetime.now().isoformat(),
            'n_samples': len(X_prod),
            'threshold': threshold,
            'roc_auc': float(roc_auc),
            'precision': float(precision),
            'recall': float(recall),
            'baseline_roc_auc': float(baseline_roc_auc),
            'roc_auc_delta': float(roc_auc_delta),
            'performance_degraded': bool(performance_degraded)
        }
        
        # Print results
        print("\n" + "-" * 80)
        print("PERFORMANCE METRICS")
        print("-" * 80)
        print(f"ROC-AUC:           {roc_auc:.4f}")
        print(f"Precision:         {precision:.4f}")
        print(f"Recall:            {recall:.4f}")
        print(f"\nBaseline ROC-AUC:  {baseline_roc_auc:.4f}")
        print(f"Delta:             {roc_auc_delta:+.4f}")
        
        if performance_degraded:
            print(f"\n⚠ WARNING: Performance degraded by {abs(roc_auc_delta):.2%}")
            print("Recommended action: Retrain model immediately")
        else:
            print(f"\n✓ Performance within acceptable range")
        
        print("-" * 80)
        
        print("\n" + "=" * 80)
        print("EVALUATION COMPLETE")
        print("=" * 80 + "\n")
        
        return results
    
    def generate_monitoring_report(
        self,
        production_df: pd.DataFrame,
        output_path: str = "./logs/monitoring_report.json"
    ) -> Dict:
        """
        Generate comprehensive monitoring report.
        
        Args:
            production_df: Production data batch
            output_path: Path to save JSON report
        
        Returns:
            Complete monitoring report
        """
        print("\n" + "=" * 80)
        print("GENERATING MONITORING REPORT")
        print("=" * 80)
        
        # Detect drift
        drift_results = self.detect_feature_drift(production_df)
        
        # Evaluate performance (if labels available)
        if 'isFraud' in production_df.columns:
            performance_results = self.evaluate_performance(production_df)
        else:
            performance_results = {
                'note': 'Ground truth labels not available'
            }
        
        # Compile report
        report = {
            'report_timestamp': datetime.now().isoformat(),
            'model_version': self.metadata.get('model_version', 'N/A'),
            'production_batch_size': len(production_df),
            'drift_detection': drift_results,
            'performance_evaluation': performance_results,
            'alerts': []
        }
        
        # Generate alerts
        if drift_results['drift_detected']:
            report['alerts'].append({
                'type': 'FEATURE_DRIFT',
                'severity': 'WARNING',
                'message': 'Feature drift detected in production data',
                'action': 'Investigate and consider model retraining'
            })
        
        if performance_results.get('performance_degraded', False):
            report['alerts'].append({
                'type': 'PERFORMANCE_DEGRADATION',
                'severity': 'CRITICAL',
                'message': f"ROC-AUC dropped by {abs(performance_results['roc_auc_delta']):.2%}",
                'action': 'Retrain model immediately'
            })
        
        # Save report
        import os
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"\n✓ Report saved: {output_path}")
        
        # Print summary
        print("\n" + "=" * 80)
        print("MONITORING REPORT SUMMARY")
        print("=" * 80)
        print(f"\nTimestamp: {report['report_timestamp']}")
        print(f"Model Version: {report['model_version']}")
        print(f"Batch Size: {report['production_batch_size']:,}")
        print(f"\nDrift Detected: {'YES ⚠' if drift_results['drift_detected'] else 'NO ✓'}")
        
        if 'roc_auc' in performance_results:
            print(f"ROC-AUC: {performance_results['roc_auc']:.4f}")
            print(f"Performance Degraded: {'YES ⚠' if performance_results.get('performance_degraded') else 'NO ✓'}")
        
        print(f"\nAlerts: {len(report['alerts'])}")
        for alert in report['alerts']:
            print(f"  - [{alert['severity']}] {alert['message']}")
        
        print("\n" + "=" * 80 + "\n")
        
        return report


def simulate_production_batch(
    reference_df: pd.DataFrame,
    batch_size: int = 1000,
    drift_simulation: str = 'none'
) -> pd.DataFrame:
    """
    Simulate a production data batch for testing.
    
    Args:
        reference_df: Reference training data
        batch_size: Size of production batch
        drift_simulation: 'none', 'moderate', or 'severe'
    
    Returns:
        Simulated production batch
    """
    # Sample from reference data
    production_batch = reference_df.sample(n=batch_size, replace=True).copy()
    
    # Simulate drift
    if drift_simulation == 'moderate':
        # Increase transaction amounts by 20%
        production_batch['TransactionAmt'] *= 1.2
        
    elif drift_simulation == 'severe':
        # Increase transaction amounts by 50%
        production_batch['TransactionAmt'] *= 1.5
        # Change fraud rate
        production_batch['isFraud'] = np.random.choice([0, 1], batch_size, p=[0.90, 0.10])
    
    return production_batch


def main():
    """Execute monitoring pipeline."""
    
    print("\n" + "=" * 80)
    print("SENTINEL PRODUCTION MONITORING - DEMO")
    print("=" * 80)
    
    # Initialize monitor
    monitor = SentinelMonitor(
        model_path="./models/sentinel_fraud_model.pkl",
        reference_data_path="./data/processed/train_engineered.pkl",
        metadata_path="./models/model_metadata.json"
    )
    
    # Load reference data for simulation
    reference_df = pd.read_pickle("./data/processed/train_engineered.pkl")
    
    # Scenario 1: No drift
    print("\n" + "=" * 80)
    print("SCENARIO 1: Normal Production Data (No Drift)")
    print("=" * 80)
    
    normal_batch = simulate_production_batch(reference_df, batch_size=1000, drift_simulation='none')
    report1 = monitor.generate_monitoring_report(normal_batch, "./logs/monitoring_report_normal.json")
    
    # Scenario 2: Moderate drift
    print("\n" + "=" * 80)
    print("SCENARIO 2: Production Data with Moderate Drift")
    print("=" * 80)
    
    drift_batch = simulate_production_batch(reference_df, batch_size=1000, drift_simulation='moderate')
    report2 = monitor.generate_monitoring_report(drift_batch, "./logs/monitoring_report_drift.json")
    
    # Scenario 3: Severe drift
    print("\n" + "=" * 80)
    print("SCENARIO 3: Production Data with Severe Drift")
    print("=" * 80)
    
    severe_batch = simulate_production_batch(reference_df, batch_size=1000, drift_simulation='severe')
    report3 = monitor.generate_monitoring_report(severe_batch, "./logs/monitoring_report_severe.json")
    
    print("\n" + "=" * 80)
    print("MONITORING DEMO COMPLETE")
    print("=" * 80)
    print("\n📊 Generated Reports:")
    print("  - ./logs/monitoring_report_normal.json")
    print("  - ./logs/monitoring_report_drift.json")
    print("  - ./logs/monitoring_report_severe.json")
    print("\n" + "=" * 80 + "\n")


if __name__ == "__main__":
    main()

"""
Deepchecks Monitoring Module

This module implements data drift detection, model performance monitoring,
and automated reporting using Deepchecks.
"""

import logging
from typing import Dict, List, Optional, Tuple, Union, Any
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
import requests
import json

from deepchecks.tabular import Dataset
from deepchecks.tabular.suites import full_suite
from deepchecks.tabular.checks import TrainTestFeatureDrift as DriftCheck
from deepchecks.tabular.checks import RegressionSystematicError as PerformanceCheck

# Import custom modules
from src.monitoring.custom_metrics import EnergyBusinessMetrics
from src.monitoring.alerting import AlertManager, create_default_energy_alert_rules

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DeepchecksMonitor:
    """
    Monitoring class using Deepchecks for data drift and model performance tracking.
    """

    def __init__(self, reference_data: pd.DataFrame, target_column: str = 'total_load_actual'):
        """
        Initialize the monitor with reference data.

        Args:
            reference_data (pd.DataFrame): Reference dataset for comparison.
            target_column (str): Name of the target column.
        """
        self.reference_data = reference_data.copy()
        self.target_column = target_column

        # Create Deepchecks Dataset objects
        self.reference_dataset = self._create_dataset(self.reference_data, "reference")

        logger.info("Deepchecks monitor initialized successfully")

    def _create_dataset(self, data: pd.DataFrame, dataset_name: str) -> Dataset:
        """
        Create a Deepchecks Dataset object.

        Args:
            data (pd.DataFrame): Input data.
            dataset_name (str): Name for the dataset.

        Returns:
            Dataset: Deepchecks Dataset object.
        """
        # Identify column types
        cat_features = []
        datetime_features = []

        for col in data.columns:
            if col == self.target_column:
                continue
            elif data[col].dtype == 'object' or data[col].dtype.name == 'category':
                cat_features.append(col)
            elif pd.api.types.is_datetime64_any_dtype(data[col]):
                datetime_features.append(col)

        # Create Dataset
        dataset = Dataset(
            data,
            label=self.target_column,
            cat_features=cat_features
        )

        return dataset

    def detect_data_drift(self, current_data: pd.DataFrame, report_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Detect data drift between reference and current data.

        Args:
            current_data (pd.DataFrame): Current dataset to compare.
            report_path (Optional[str]): Path to save the report.

        Returns:
            Dict[str, Any]: Drift detection results.
        """
        logger.info("Detecting data drift...")

        try:
            # Create current dataset
            current_dataset = self._create_dataset(current_data, "current")

            # Run drift check
            drift_check = DriftCheck()
            drift_result = drift_check.run(self.reference_dataset, current_dataset)

            # Extract drift results
            drift_summary = drift_result.value

            # Determine if drift is detected (simplified logic)
            drift_detected = False
            drift_score = 0.0

            if hasattr(drift_summary, 'drift_score'):
                drift_score = drift_summary.drift_score
                drift_detected = drift_score > 0.1  # Threshold for drift detection
            else:
                # Fallback: check individual feature drifts
                for feature_name, feature_drift in drift_summary.items():
                    if hasattr(feature_drift, 'drift_score'):
                        feature_score = feature_drift.drift_score
                        drift_score = max(drift_score, feature_score)
                        if feature_score > 0.1:
                            drift_detected = True

            # Save report if path provided
            if report_path:
                drift_result.save_as_html(report_path)
                logger.info(f"Data drift report saved to {report_path}")

            results = {
                'drift_detected': drift_detected,
                'drift_score': drift_score,
                'report_path': report_path,
                'timestamp': datetime.now().isoformat()
            }

        except Exception as e:
            logger.warning(f"Data drift detection failed: {str(e)}. Using simplified approach.")
            # Simplified drift detection based on basic statistics
            drift_detected = False
            drift_score = 0.0

            # Check for basic statistical differences in numerical columns
            numerical_cols = current_data.select_dtypes(include=[np.number]).columns
            for col in numerical_cols[:5]:  # Check first 5 columns
                if col in self.reference_data.columns and col in current_data.columns:
                    ref_mean = self.reference_data[col].mean()
                    curr_mean = current_data[col].mean()
                    if abs(ref_mean - curr_mean) > 0.1 * abs(ref_mean):  # 10% difference threshold
                        drift_detected = True
                        drift_score += 0.1

            results = {
                'drift_detected': drift_detected,
                'drift_score': drift_score,
                'report_path': report_path,
                'timestamp': datetime.now().isoformat(),
                'method': 'simplified'
            }

        logger.info(f"Data drift detection completed. Drift detected: {results['drift_detected']}")
        return results

    def detect_data_drift_http(self, data_url: str, api_key: Optional[str] = None,
                             report_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Detect data drift by loading data from HTTP endpoint.

        Args:
            data_url (str): URL to fetch current data from.
            api_key (Optional[str]): API key for authentication.
            report_path (Optional[str]): Path to save the report.

        Returns:
            Dict[str, Any]: Drift detection results.
        """
        logger.info(f"Detecting data drift from HTTP endpoint: {data_url}")

        try:
            # Prepare headers
            headers = {'Content-Type': 'application/json'}
            if api_key:
                headers['Authorization'] = f'Bearer {api_key}'

            # Fetch data from HTTP endpoint
            response = requests.get(data_url, headers=headers, timeout=30)
            response.raise_for_status()

            # Parse JSON data
            data_json = response.json()

            # Convert to DataFrame
            if isinstance(data_json, list):
                current_data = pd.DataFrame(data_json)
            elif isinstance(data_json, dict) and 'data' in data_json:
                current_data = pd.DataFrame(data_json['data'])
            else:
                raise ValueError("Invalid data format from HTTP endpoint")

            # Process datetime column if present
            if 'time' in current_data.columns:
                current_data['time'] = pd.to_datetime(current_data['time'])
                current_data.set_index('time', inplace=True)

            # Run drift detection
            return self.detect_data_drift(current_data, report_path)

        except requests.RequestException as e:
            logger.error(f"HTTP request failed: {str(e)}")
            raise Exception(f"Failed to fetch data from {data_url}: {str(e)}")
        except Exception as e:
            logger.error(f"Data drift detection from HTTP failed: {str(e)}")
            raise Exception(f"Failed to process data from HTTP endpoint: {str(e)}")

    def monitor_model_performance(self, y_true: pd.Series, y_pred: pd.Series,
                                report_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Monitor model performance using Deepchecks performance checks.

        Args:
            y_true (pd.Series): True values.
            y_pred (pd.Series): Predicted values.
            report_path (Optional[str]): Path to save the report.

        Returns:
            Dict[str, Any]: Performance monitoring results.
        """
        logger.info("Monitoring model performance...")

        try:
            # Create dataset with predictions
            performance_data = self.reference_data.copy()
            performance_data['prediction'] = y_pred.values[:len(performance_data)]

            # Create dataset for performance monitoring
            perf_dataset = self._create_dataset(performance_data, "performance")

            # Run performance check
            performance_check = PerformanceCheck()
            perf_result = performance_check.run(perf_dataset)

            # Extract performance metrics
            perf_summary = perf_result.value

            # Extract key metrics (simplified)
            mae = perf_summary.get('MAE', 0) if isinstance(perf_summary, dict) else 0
            rmse = perf_summary.get('RMSE', 0) if isinstance(perf_summary, dict) else 0
            r2_score = perf_summary.get('R2', 0) if isinstance(perf_summary, dict) else 0

            # Save report if path provided
            if report_path:
                perf_result.save_as_html(report_path)
                logger.info(f"Performance report saved to {report_path}")

            results = {
                'mae': mae,
                'rmse': rmse,
                'r2_score': r2_score,
                'mean_error': 0,  # Placeholder
                'report_path': report_path,
                'timestamp': datetime.now().isoformat()
            }

        except Exception as e:
            logger.warning(f"Performance monitoring failed: {str(e)}. Using simplified approach.")
            # Simplified performance calculation
            from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

            mae = mean_absolute_error(y_true, y_pred)
            rmse = np.sqrt(mean_squared_error(y_true, y_pred))
            r2 = r2_score(y_true, y_pred)

            results = {
                'mae': mae,
                'rmse': rmse,
                'r2_score': r2,
                'mean_error': np.mean(y_pred - y_true),
                'report_path': report_path,
                'timestamp': datetime.now().isoformat(),
                'method': 'simplified'
            }

        logger.info(f"Model performance monitoring completed. MAE: {results['mae']:.4f}")
        return results

    def create_data_quality_report(self, data: pd.DataFrame, report_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Create data quality report for dataset validation.

        Args:
            data (pd.DataFrame): Dataset to analyze.
            report_path (Optional[str]): Path to save the report.

        Returns:
            Dict[str, Any]: Data quality assessment results.
        """
        logger.info("Creating data quality report...")

        try:
            # Create dataset for quality check
            quality_dataset = self._create_dataset(data, "quality_check")

            # Run data quality suite
            quality_suite = full_suite()
            quality_result = quality_suite.run(quality_dataset)

            # Extract quality metrics (simplified)
            quality_summary = quality_result.value

            # Calculate basic quality score
            missing_values = {}
            quality_score = 1.0

            for col in data.columns:
                missing_pct = data[col].isnull().sum() / len(data)
                if missing_pct > 0:
                    missing_values[col] = missing_pct
                    quality_score -= missing_pct * 0.1  # Penalize for missing values

            # Save report if path provided
            if report_path:
                quality_result.save_as_html(report_path)
                logger.info(f"Data quality report saved to {report_path}")

            quality_results = {
                'missing_values': missing_values,
                'data_quality_score': max(0, quality_score),
                'report_path': report_path,
                'timestamp': datetime.now().isoformat()
            }

        except Exception as e:
            logger.warning(f"Data quality report failed: {str(e)}. Using simplified approach.")
            # Simplified data quality check
            missing_values = {}
            quality_score = 1.0

            for col in data.columns:
                missing_pct = data[col].isnull().sum() / len(data)
                if missing_pct > 0:
                    missing_values[col] = missing_pct
                    quality_score -= missing_pct * 0.1  # Penalize for missing values

            quality_results = {
                'missing_values': missing_values,
                'data_quality_score': max(0, quality_score),
                'report_path': report_path,
                'timestamp': datetime.now().isoformat(),
                'method': 'simplified'
            }

        logger.info("Data quality report completed.")
        return quality_results

    def check_alerts(self, drift_results: Dict[str, Any], performance_results: Dict[str, Any],
                   thresholds: Dict[str, float]) -> List[str]:
        """
        Check for alerts based on monitoring results and thresholds.

        Args:
            drift_results (Dict[str, Any]): Data drift detection results.
            performance_results (Dict[str, Any]): Model performance results.
            thresholds (Dict[str, float]): Alert thresholds.

        Returns:
            List[str]: List of triggered alerts.
        """
        alerts = []

        # Check data drift
        if drift_results.get('drift_detected', False):
            alerts.append(f"Data drift detected! Drift score: {drift_results.get('drift_score', 0):.3f}")

        # Check performance degradation
        if performance_results.get('mae', 0) > thresholds.get('max_mae', float('inf')):
            alerts.append(f"Model performance degraded! MAE: {performance_results.get('mae', 0):.4f}")

        if performance_results.get('r2_score', 1) < thresholds.get('min_r2', -float('inf')):
            alerts.append(f"Model R² score below threshold! R²: {performance_results.get('r2_score', 0):.4f}")

        return alerts


class MonitoringPipeline:
    """
    Complete monitoring pipeline integrating all Deepchecks components.
    """

    def __init__(self, reference_data_path: str, target_column: str = 'total_load_actual'):
        """
        Initialize the monitoring pipeline.

        Args:
            reference_data_path (str): Path to reference dataset.
            target_column (str): Target column name.
        """
        self.reference_data_path = reference_data_path
        self.target_column = target_column
        self.monitor = None
        self.alert_thresholds = {
            'max_mae': 1000,  # Adjust based on your data scale
            'min_r2': 0.7,
            'drift_threshold': 0.05
        }

    def initialize_monitoring(self) -> None:
        """
        Initialize the monitoring system with reference data.
        """
        logger.info("Initializing monitoring system...")

        # Load reference data
        reference_data = pd.read_csv(self.reference_data_path)
        if 'time' in reference_data.columns:
            reference_data['time'] = pd.to_datetime(reference_data['time'])
            reference_data.set_index('time', inplace=True)

        # Initialize monitor
        self.monitor = DeepchecksMonitor(reference_data, self.target_column)

        logger.info("Monitoring system initialized successfully")

    def run_monitoring_cycle(self, current_data: pd.DataFrame, predictions: Optional[pd.Series] = None,
                            output_dir: str = "reports/monitoring") -> Dict[str, Any]:
        """
        Run a complete monitoring cycle.

        Args:
            current_data (pd.DataFrame): Current data to monitor.
            predictions (Optional[pd.Series]): Model predictions for performance monitoring.
            output_dir (str): Directory to save monitoring reports.

        Returns:
            Dict[str, Any]: Monitoring cycle results.
        """
        logger.info("Running monitoring cycle...")

        # Ensure output directory exists
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        results = {
            'timestamp': datetime.now().isoformat(),
            'data_quality': {},
            'data_drift': {},
            'model_performance': {},
            'alerts': []
        }

        # Data quality check
        quality_report_path = f"{output_dir}/data_quality_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
        results['data_quality'] = self.monitor.create_data_quality_report(
            current_data, quality_report_path
        )

        # Data drift detection
        drift_report_path = f"{output_dir}/data_drift_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
        results['data_drift'] = self.monitor.detect_data_drift(
            current_data, drift_report_path
        )

        # Model performance monitoring (if predictions available)
        if predictions is not None and len(predictions) > 0:
            # Get corresponding true values
            true_values = current_data[self.target_column].tail(len(predictions))

            perf_report_path = f"{output_dir}/model_performance_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
            results['model_performance'] = self.monitor.monitor_model_performance(
                true_values, predictions, perf_report_path
            )

        # Check for alerts
        results['alerts'] = self.monitor.check_alerts(
            results['data_drift'], results['model_performance'], self.alert_thresholds
        )

        logger.info(f"Monitoring cycle completed. Alerts triggered: {len(results['alerts'])}")
        return results


if __name__ == "__main__":
    # Example usage
    import argparse

    parser = argparse.ArgumentParser(description='Run Deepchecks monitoring')
    parser.add_argument('--reference-data', type=str, default='data/processed/processed_energy_weather.csv',
                       help='Path to reference dataset')
    parser.add_argument('--current-data', type=str, default='data/processed/current_batch.csv',
                       help='Path to current dataset')
    parser.add_argument('--predictions', type=str, help='Path to predictions CSV')
    parser.add_argument('--output-dir', type=str, default='reports/monitoring',
                       help='Output directory for reports')

    args = parser.parse_args()

    # Initialize monitoring
    monitor_pipeline = MonitoringPipeline(args.reference_data)
    monitor_pipeline.initialize_monitoring()

    # Load current data
    current_data = pd.read_csv(args.current_data)
    if 'time' in current_data.columns:
        current_data['time'] = pd.to_datetime(current_data['time'])
        current_data.set_index('time', inplace=True)

    # Load predictions if available
    predictions = None
    if args.predictions:
        pred_df = pd.read_csv(args.predictions)
        predictions = pred_df['prediction'] if 'prediction' in pred_df.columns else None

    # Run monitoring cycle
    results = monitor_pipeline.run_monitoring_cycle(current_data, predictions, args.output_dir)

    print("Monitoring cycle completed.")
    print(f"Data drift detected: {results['data_drift'].get('drift_detected', False)}")
    print(f"Alerts: {results['alerts']}")
    if results['model_performance']:
        print(f"MAE: {results['model_performance'].get('mae', 0):.4f}")

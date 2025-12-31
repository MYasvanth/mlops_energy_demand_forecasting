"""
Evidently Monitoring Module

This module implements data drift detection, model performance monitoring,
and automated reporting using Evidently AI with robust error handling.
"""

import logging
from typing import Dict, List, Optional, Tuple, Union, Any
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
import requests
import json

# Configure logging FIRST
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
# Import Evidently components with robust fallback handling
try:
    from evidently.pipeline import Pipeline
    from evidently.metric_preset import DataDriftPreset, DataQualityPreset, RegressionPreset
    from evidently import ColumnMapping
    EVIDENTLY_AVAILABLE = True
    logger.info("Evidently AI library loaded successfully")
except (ImportError, RuntimeError) as e:
    logger.warning(f"Evidently not available: {e}. Using simplified monitoring.")
    EVIDENTLY_AVAILABLE = False
    Pipeline = None
    DataDriftPreset = None
    DataQualityPreset = None
    RegressionPreset = None
    ColumnMapping = None

# Import custom modules
from src.monitoring.custom_metrics import EnergyBusinessMetrics
from src.monitoring.alerting import AlertManager, create_default_energy_alert_rules

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EvidentlyMonitor:
    """
    Monitoring class using Evidently for data drift and model performance tracking.
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
        self.column_mapping = self._create_column_mapping()
        # Initialize workspace (will be None for simplified monitoring)
        self.workspace = None
        # Alert thresholds for monitoring
        self.alert_thresholds = {
            'max_mae': 1000,  # Adjust based on your data scale
            'min_r2': 0.7,
            'drift_threshold': 0.05
        }

    def _create_column_mapping(self) -> Optional[ColumnMapping]:
        """
        Create column mapping for Evidently reports.

        Returns:
            Optional[ColumnMapping]: Column mapping configuration or None if Evidently not available.
        """
        if not EVIDENTLY_AVAILABLE or ColumnMapping is None:
            return None

        # Define column types
        numerical_features = []
        categorical_features = []
        datetime_features = []

        for col in self.reference_data.columns:
            if col == self.target_column:
                continue
            elif self.reference_data[col].dtype in ['int64', 'float64']:
                if col.endswith('_sin') or col.endswith('_cos') or 'lag' in col:
                    numerical_features.append(col)
                else:
                    numerical_features.append(col)
            elif self.reference_data[col].dtype == 'object':
                categorical_features.append(col)
            elif pd.api.types.is_datetime64_any_dtype(self.reference_data[col]):
                datetime_features.append(col)

        return ColumnMapping(
            target=self.target_column,
            prediction=None,  # Will be set when monitoring predictions
            numerical_features=numerical_features,
            categorical_features=categorical_features,
            datetime_features=datetime_features
        )

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

        if EVIDENTLY_AVAILABLE and Pipeline is not None and DataDriftPreset is not None:
            try:
                # Create data drift pipeline
                drift_pipeline = Pipeline([DataDriftPreset()])
                drift_result = drift_pipeline.execute(reference_data=self.reference_data,
                                                    current_data=current_data)

                # Extract drift metrics
                drift_metrics = drift_result.as_dict()

                # Save report if path provided
                if report_path:
                    drift_result.save_html(report_path)
                    logger.info(f"Data drift report saved to {report_path}")

                # Check for significant drift
                drift_detected = any(
                    metric.get('drift_detected', False)
                    for metric in drift_metrics.get('metrics', [])
                    if isinstance(metric, dict) and 'drift_detected' in metric
                )

                results = {
                    'drift_detected': drift_detected,
                    'drift_score': drift_metrics.get('metrics', [{}])[0].get('drift_score', 0),
                    'report_path': report_path,
                    'timestamp': datetime.now().isoformat()
                }

            except Exception as e:
                logger.warning(f"Data drift detection failed: {str(e)}. Using simplified approach.")
                # Simplified drift detection based on basic statistics
                drift_detected = False
                drift_score = 0.0

                # Check for basic statistical differences in numerical columns
                if self.column_mapping and hasattr(self.column_mapping, 'numerical_features'):
                    for col in self.column_mapping.numerical_features[:5]:  # Check first 5 columns
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
        else:
            logger.warning("Evidently not available. Using simplified drift detection.")
            # Simplified drift detection based on basic statistics
            drift_detected = False
            drift_score = 0.0

            # Check for basic statistical differences in numerical columns
            numerical_cols = [col for col in self.reference_data.select_dtypes(include=[np.number]).columns
                            if col != self.target_column][:5]  # Check first 5 numerical columns

            for col in numerical_cols:
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
                'method': 'fallback'
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
        Monitor model performance using Evidently regression metrics.

        Args:
            y_true (pd.Series): True values.
            y_pred (pd.Series): Predicted values.
            report_path (Optional[str]): Path to save the report.

        Returns:
            Dict[str, Any]: Performance monitoring results.
        """
        logger.info("Monitoring model performance...")

        if EVIDENTLY_AVAILABLE and Pipeline is not None and RegressionPreset is not None:
            try:
                # Create performance DataFrame
                performance_data = pd.DataFrame({
                    'target': y_true,
                    'prediction': y_pred
                })

                # Create regression performance pipeline
                regression_pipeline = Pipeline([RegressionPreset()])
                regression_result = regression_pipeline.execute(reference_data=performance_data,
                                                               current_data=performance_data)

                # Extract performance metrics
                performance_metrics = regression_result.as_dict()

                # Save report if path provided
                if report_path:
                    regression_result.save_html(report_path)
                    logger.info(f"Performance report saved to {report_path}")

                # Extract key metrics
                metrics_dict = {}
                for metric in performance_metrics.get('metrics', []):
                    if isinstance(metric, dict) and 'metric' in metric:
                        metric_name = metric['metric']
                        if 'value' in metric:
                            metrics_dict[metric_name] = metric['value']

                results = {
                    'mae': metrics_dict.get('MAE', 0),
                    'rmse': metrics_dict.get('RMSE', 0),
                    'r2_score': metrics_dict.get('R2Score', 0),
                    'mean_error': metrics_dict.get('MeanError', 0),
                    'report_path': report_path,
                    'timestamp': datetime.now().isoformat()
                }

            except Exception as e:
                logger.warning(f"Model performance monitoring failed: {str(e)}. Using simplified approach.")
                # Simplified performance calculation
                mae = np.mean(np.abs(y_true - y_pred))
                rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
                r2 = 1 - (np.sum((y_true - y_pred) ** 2) / np.sum((y_true - np.mean(y_true)) ** 2))
                mean_error = np.mean(y_true - y_pred)

                results = {
                    'mae': mae,
                    'rmse': rmse,
                    'r2_score': r2,
                    'mean_error': mean_error,
                    'report_path': report_path,
                    'timestamp': datetime.now().isoformat(),
                    'method': 'simplified'
                }
        else:
            logger.warning("Evidently not available. Using simplified performance monitoring.")
            # Simplified performance calculation
            mae = np.mean(np.abs(y_true - y_pred))
            rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
            r2 = 1 - (np.sum((y_true - y_pred) ** 2) / np.sum((y_true - np.mean(y_true)) ** 2))
            mean_error = np.mean(y_true - y_pred)

            results = {
                'mae': mae,
                'rmse': rmse,
                'r2_score': r2,
                'mean_error': mean_error,
                'report_path': report_path,
                'timestamp': datetime.now().isoformat(),
                'method': 'fallback'
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

        if EVIDENTLY_AVAILABLE and Pipeline is not None and DataQualityPreset is not None:
            try:
                # Create data quality pipeline
                quality_pipeline = Pipeline([DataQualityPreset()])
                quality_result = quality_pipeline.execute(reference_data=self.reference_data,
                                                         current_data=data)

                # Extract quality metrics
                quality_metrics = quality_result.as_dict()

                # Save report if path provided
                if report_path:
                    quality_result.save_html(report_path)
                    logger.info(f"Data quality report saved to {report_path}")

                # Extract key quality indicators
                quality_results = {
                    'missing_values': quality_metrics.get('metrics', [{}])[0].get('missing_values', {}),
                    'data_quality_score': quality_metrics.get('metrics', [{}])[0].get('quality_score', 0),
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
        else:
            logger.warning("Evidently not available. Using simplified data quality check.")
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
                'method': 'fallback'
            }

        logger.info("Data quality report completed.")
        return quality_results

    def setup_monitoring_dashboard(self, dashboard_name: str = "Energy Demand Monitoring") -> str:
        """
        Set up Evidently monitoring dashboard.

        Args:
            dashboard_name (str): Name of the dashboard.

        Returns:
            str: Dashboard URL.
        """
        try:
            if self.workspace is None:
                logger.warning("Workspace not available, cannot create dashboard")
                return None

            # Create dashboard panels
            from evidently.ui.dashboards import DashboardPanel, DashboardConfig
            from evidently.ui.dashboards.monitoring import MonitoringDashboard
            from evidently.ui.dashboards.reports import ReportDashboard

            # Create monitoring dashboard
            monitoring_config = DashboardConfig(
                name=dashboard_name,
                panels=[
                    DashboardPanel(
                        title="Data Drift Overview",
                        filter=DashboardPanel.Filter(type="data_drift"),
                        size=DashboardPanel.Size.full
                    ),
                    DashboardPanel(
                        title="Data Quality Metrics",
                        filter=DashboardPanel.Filter(type="data_quality"),
                        size=DashboardPanel.Size.half
                    ),
                    DashboardPanel(
                        title="Model Performance",
                        filter=DashboardPanel.Filter(type="regression_performance"),
                        size=DashboardPanel.Size.half
                    ),
                    DashboardPanel(
                        title="Prediction Drift",
                        filter=DashboardPanel.Filter(type="prediction_drift"),
                        size=DashboardPanel.Size.half
                    )
                ]
            )

            # Add dashboard to workspace
            dashboard_id = self.workspace.add_dashboard(monitoring_config)
            dashboard_url = f"http://localhost:8000/monitoring/dashboard/{dashboard_id}"

            logger.info(f"Monitoring dashboard created: {dashboard_url}")
            return dashboard_url

        except Exception as e:
            logger.error(f"Failed to create monitoring dashboard: {str(e)}")
            return None

    def log_monitoring_results(self, results: Dict[str, Any], experiment_name: str = "monitoring") -> None:
        """
        Log monitoring results to Evidently workspace.

        Args:
            results (Dict[str, Any]): Monitoring results to log.
            experiment_name (str): Name of the monitoring experiment.
        """
        try:
            # Add results to workspace
            self.workspace.add_report(experiment_name, results)
            logger.info(f"Monitoring results logged to workspace under experiment: {experiment_name}")
        except Exception as e:
            logger.warning(f"Failed to log results to workspace: {str(e)}. Results will be logged to console only.")
            print(f"Monitoring Results for {experiment_name}:")
            print(f"  Timestamp: {results.get('timestamp', 'N/A')}")
            print(f"  Data Drift Detected: {results.get('data_drift', {}).get('drift_detected', False)}")
            print(f"  Alerts: {results.get('alerts', [])}")

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
        results['data_quality'] = self.create_data_quality_report(
            current_data, quality_report_path
        )

        # Data drift detection
        drift_report_path = f"{output_dir}/data_drift_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
        results['data_drift'] = self.detect_data_drift(
            current_data, drift_report_path
        )

        # Model performance monitoring (if predictions available)
        if predictions is not None and len(predictions) > 0:
            # Get corresponding true values
            true_values = current_data[self.target_column].tail(len(predictions))

            perf_report_path = f"{output_dir}/model_performance_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
            results['model_performance'] = self.monitor_model_performance(
                true_values, predictions, perf_report_path
            )

        # Check for alerts
        results['alerts'] = self.check_alerts(
            results['data_drift'], results['model_performance'], self.alert_thresholds
        )

        # Log results to workspace
        self.log_monitoring_results(results)

        logger.info(f"Monitoring cycle completed. Alerts triggered: {len(results['alerts'])}")
        return results


class MonitoringPipeline:
    """
    Complete monitoring pipeline integrating all Evidently components.
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
        self.monitor = EvidentlyMonitor(reference_data, self.target_column)

        # Setup dashboard
        dashboard_id = self.monitor.setup_monitoring_dashboard()
        logger.info(f"Monitoring system initialized. Dashboard ID: {dashboard_id}")

    def run_monitoring_cycle(self, current_data: pd.DataFrame, predictions: Optional[np.ndarray] = None,
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
                true_values, pd.Series(predictions), perf_report_path
            )

        # Check for alerts
        results['alerts'] = self.monitor.check_alerts(
            results['data_drift'], results['model_performance'], self.alert_thresholds
        )

        # Log results to workspace
        self.monitor.log_monitoring_results(results)

        logger.info(f"Monitoring cycle completed. Alerts triggered: {len(results['alerts'])}")
        return results


if __name__ == "__main__":
    # Example usage
    import argparse

    parser = argparse.ArgumentParser(description='Run Evidently monitoring')
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

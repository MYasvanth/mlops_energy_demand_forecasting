#!/usr/bin/env python3
"""
Monitoring Script for Energy Demand Forecasting

This script runs the Evidently monitoring pipeline for data drift detection
and model performance monitoring.
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Dict, Any, Optional

import pandas as pd

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.monitoring.evidently_monitoring import MonitoringPipeline
from src.data.ingestion import ingest_data
from src.data.preprocessing import full_preprocessing_pipeline
from src.features.feature_engineering import full_feature_engineering_pipeline


def setup_logging(log_level: str = "INFO", log_file: Optional[str] = None) -> None:
    """
    Set up logging configuration.

    Args:
        log_level (str): Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL).
        log_file (Optional[str]): Path to log file. If None, logs to console only.
    """
    numeric_level = getattr(logging, log_level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError(f"Invalid log level: {log_level}")

    logging.basicConfig(
        level=numeric_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(log_file) if log_file else logging.NullHandler()
        ]
    )


def main():
    """
    Main function to run the monitoring pipeline.
    """
    parser = argparse.ArgumentParser(description='Run Evidently monitoring for energy demand forecasting')
    parser.add_argument('--energy-path', type=str, default='data/raw/energy_dataset.csv',
                       help='Path to energy dataset')
    parser.add_argument('--weather-path', type=str, default='data/raw/weather_features.csv',
                       help='Path to weather dataset')
    parser.add_argument('--reference-data', type=str, default='data/processed/processed_energy_weather.csv',
                       help='Path to reference dataset for monitoring')
    parser.add_argument('--current-data', type=str, default='data/processed/current_batch.csv',
                       help='Path to current dataset to monitor')
    parser.add_argument('--predictions', type=str, help='Path to predictions CSV for performance monitoring')
    parser.add_argument('--output-dir', type=str, default='reports/monitoring',
                       help='Output directory for monitoring reports')
    parser.add_argument('--log-level', type=str, default='INFO',
                       help='Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)')
    parser.add_argument('--log-file', type=str, help='Path to log file')
    parser.add_argument('--target-column', type=str, default='total_load_actual',
                       help='Target column name')

    args = parser.parse_args()

    # Set up logging
    setup_logging(args.log_level, args.log_file)
    logger = logging.getLogger(__name__)

    try:
        # Check if reference data exists, if not create it
        reference_path = Path(args.reference_data)
        if not reference_path.exists():
            logger.info("Reference data not found. Creating reference dataset...")

            # Load and preprocess data
            raw_data = ingest_data(args.energy_path, args.weather_path)
            processed_data = full_preprocessing_pipeline(raw_data['energy'], raw_data['weather'])
            reference_data = full_feature_engineering_pipeline(processed_data)

            # Save reference data
            reference_path.parent.mkdir(parents=True, exist_ok=True)
            reference_data.to_csv(reference_path, index=False)
            logger.info(f"Reference data saved to {reference_path}")

        # Check if current data exists, if not create a sample
        current_path = Path(args.current_data)
        if not current_path.exists():
            logger.info("Current data not found. Creating sample current dataset...")

            # Load reference data and create a sample current batch
            reference_data = pd.read_csv(reference_path)
            # Take last 10% as current data
            split_idx = int(len(reference_data) * 0.9)
            current_data = reference_data.iloc[split_idx:].copy()

            # Save current data
            current_path.parent.mkdir(parents=True, exist_ok=True)
            current_data.to_csv(current_path, index=False)
            logger.info(f"Current data saved to {current_path}")

        # Initialize monitoring pipeline
        logger.info("Initializing monitoring pipeline...")
        monitor_pipeline = MonitoringPipeline(str(reference_path), args.target_column)
        monitor_pipeline.initialize_monitoring()

        # Load current data
        logger.info("Loading current data...")
        current_data = pd.read_csv(current_path)
        if 'time' in current_data.columns:
            current_data['time'] = pd.to_datetime(current_data['time'])
            current_data.set_index('time', inplace=True)

        # Load predictions if available
        predictions = None
        if args.predictions and Path(args.predictions).exists():
            logger.info("Loading predictions...")
            pred_df = pd.read_csv(args.predictions)
            predictions = pred_df['prediction'] if 'prediction' in pred_df.columns else None

        # Run monitoring cycle
        logger.info("Running monitoring cycle...")
        results = monitor_pipeline.run_monitoring_cycle(
            current_data, predictions, args.output_dir
        )

        # Print results
        print("\n" + "="*60)
        print("MONITORING RESULTS")
        print("="*60)
        print(f"Timestamp: {results['timestamp']}")
        print(f"Data Quality Report: {results['data_quality'].get('report_path', 'N/A')}")
        print(f"Data Drift Report: {results['data_drift'].get('report_path', 'N/A')}")
        if results['model_performance']:
            print(f"Performance Report: {results['model_performance'].get('report_path', 'N/A')}")

        print("\nData Drift Detection:")
        print(f"  - Drift Detected: {results['data_drift'].get('drift_detected', False)}")
        print(f"  - Drift Score: {results['data_drift'].get('drift_score', 0):.4f}")

        if results['model_performance']:
            print("\nModel Performance:")
            print(f"  - MAE: {results['model_performance'].get('mae', 0):.4f}")
            print(f"  - RMSE: {results['model_performance'].get('rmse', 0):.4f}")
            print(f"  - R² Score: {results['model_performance'].get('r2_score', 0):.4f}")

        if results['alerts']:
            print("\nAlerts:")
            for alert in results['alerts']:
                print(f"  - {alert}")
        else:
            print("\nNo alerts triggered.")

        print("="*60)

        logger.info("Monitoring completed successfully")

    except Exception as e:
        logger.error(f"Monitoring failed: {str(e)}")
        print(f"\nERROR: {str(e)}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()

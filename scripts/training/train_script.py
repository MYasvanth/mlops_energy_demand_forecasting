#!/usr/bin/env python3
"""
Training Script for Energy Demand Forecasting

This script provides a command-line interface to run the training pipeline
with configurable parameters and options.
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Dict, Any, Optional

import yaml
import pandas as pd
from prefect import flow

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.config.config_loader import load_app_config
from src.data.ingestion import ingest_data
from src.data.preprocessing import full_preprocessing_pipeline
from src.features.feature_engineering import full_feature_engineering_pipeline
from src.models.train import train_multiple_models


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


def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load configuration from YAML file.

    Args:
        config_path (str): Path to configuration file.

    Returns:
        Dict[str, Any]: Configuration dictionary.
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def validate_paths(energy_path: str, weather_path: str, config_path: str) -> None:
    """
    Validate that required file paths exist.

    Args:
        energy_path (str): Path to energy dataset.
        weather_path (str): Path to weather dataset.
        config_path (str): Path to configuration file.

    Raises:
        FileNotFoundError: If any required file is missing.
    """
    paths_to_check = [
        ("Energy dataset", energy_path),
        ("Weather dataset", weather_path),
        ("Configuration file", config_path)
    ]

    for name, path in paths_to_check:
        if not Path(path).exists():
            raise FileNotFoundError(f"{name} not found at: {path}")


def run_training_pipeline(
    energy_path: str,
    weather_path: str,
    config_path: str,
    target_column: str = "total_load_actual",
    output_dir: str = "models/production",
    use_prefect: bool = False
) -> Dict[str, Any]:
    """
    Run the complete training pipeline.

    Args:
        energy_path (str): Path to energy dataset.
        weather_path (str): Path to weather dataset.
        config_path (str): Path to configuration file.
        target_column (str): Target column for forecasting.
        output_dir (str): Directory to save trained models.
        use_prefect (bool): Whether to use Prefect orchestration.

    Returns:
        Dict[str, Any]: Training results.
    """
    logger = logging.getLogger(__name__)

    try:
        # Load configuration
        logger.info(f"Loading configuration from {config_path}")
        config = load_config(config_path)

        # Load and preprocess data
        logger.info("Loading and preprocessing data...")
        raw_data = ingest_data(energy_path, weather_path)
        processed_df = full_preprocessing_pipeline(raw_data['energy'], raw_data['weather'])

        # Create features
        logger.info("Creating features...")
        feature_df = full_feature_engineering_pipeline(processed_df)

        # Train models
        logger.info("Training models...")
        trained_models = train_multiple_models(feature_df, target_column, models=['arima', 'prophet', 'lstm'])

        # Extract evaluation results
        evaluation_results = {model: result['metrics'] for model, result in trained_models.items()}

        results = {
            'status': 'success',
            'models_trained': list(trained_models.keys()),
            'evaluation_results': evaluation_results,
            'output_dir': output_dir
        }

        logger.info("Training pipeline completed successfully")
        return results

    except Exception as e:
        logger.error(f"Training pipeline failed: {str(e)}")
        raise


def main():
    """Main entry point for the training script."""
    parser = argparse.ArgumentParser(
        description="Train energy demand forecasting models",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Data paths
    parser.add_argument(
        '--energy-path',
        type=str,
        default='data/raw/energy_dataset.csv',
        help='Path to energy dataset CSV file'
    )
    parser.add_argument(
        '--weather-path',
        type=str,
        default='data/raw/weather_features.csv',
        help='Path to weather dataset CSV file'
    )

    # Configuration
    parser.add_argument(
        '--config',
        type=str,
        default='configs/model/model_config.yaml',
        help='Path to model configuration YAML file'
    )

    # Training parameters
    parser.add_argument(
        '--target-column',
        type=str,
        default='total_load_actual',
        help='Target column for forecasting'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='models/production',
        help='Directory to save trained models'
    )

    # Orchestration options
    parser.add_argument(
        '--use-prefect',
        action='store_true',
        help='Use Prefect for orchestration instead of direct execution'
    )
    parser.add_argument(
        '--prefect-config',
        type=str,
        default='configs/prefect/prefect_config.yaml',
        help='Path to Prefect configuration file (used with --use-prefect)'
    )

    # Logging options
    parser.add_argument(
        '--log-level',
        type=str,
        default='INFO',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
        help='Logging level'
    )
    parser.add_argument(
        '--log-file',
        type=str,
        help='Path to log file (optional, logs to console if not specified)'
    )

    # Validation
    parser.add_argument(
        '--validate-only',
        action='store_true',
        help='Only validate inputs without running training'
    )

    args = parser.parse_args()

    # Set up logging
    setup_logging(args.log_level, args.log_file)
    logger = logging.getLogger(__name__)

    try:
        # Validate paths
        logger.info("Validating input paths...")
        validate_paths(args.energy_path, args.weather_path, args.config)

        if args.validate_only:
            logger.info("Validation successful. Exiting.")
            return

        if args.use_prefect:
            # Use Prefect orchestration
            logger.info("Using Prefect orchestration...")
            prefect_config = load_config(args.prefect_config)

            results = energy_demand_orchestration_flow(
                energy_path=args.energy_path,
                weather_path=args.weather_path,
                target_column=args.target_column,
                enable_notifications=prefect_config.get('notifications', {}).get('enabled', True)
            )
        else:
            # Run training pipeline directly
            logger.info("Running training pipeline directly...")
            results = run_training_pipeline(
                energy_path=args.energy_path,
                weather_path=args.weather_path,
                config_path=args.config,
                target_column=args.target_column,
                output_dir=args.output_dir,
                use_prefect=False
            )

        # Print results
        print("\n" + "="*50)
        print("TRAINING RESULTS")
        print("="*50)
        print(f"Status: {results['status']}")
        if results['status'] == 'success':
            print(f"Models trained: {', '.join(results['models_trained'])}")
            print(f"Output directory: {results['output_dir']}")
            print("\nEvaluation Results:")
            for model, metrics in results['evaluation_results'].items():
                print(f"  {model}:")
                for metric, value in metrics.items():
                    print(f"    {metric}: {value:.4f}")
        print("="*50)

        logger.info("Training script completed successfully")

    except Exception as e:
        logger.error(f"Training script failed: {str(e)}")
        print(f"\nERROR: {str(e)}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()

"""
ZenML Training Pipeline for Energy Demand Forecasting

This module defines a ZenML pipeline for the complete ML training workflow,
including data ingestion, preprocessing, feature engineering, and model training.
Integrated with MLflow for experiment tracking and Optuna for hyperparameter tuning.
"""

import logging
from typing import Dict, Any
import pandas as pd
import mlflow
import mlflow.sklearn
from zenml import pipeline, step
from zenml.client import Client
try:
    from zenml.integrations.mlflow.mlflow_step_decorator import enable_mlflow
except ImportError:
    # Fallback for older ZenML versions
    def enable_mlflow(func):
        return func

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@step
def ingest_data_step(energy_path: str, weather_path: str) -> Dict[str, pd.DataFrame]:
    """
    ZenML step for data ingestion.

    Args:
        energy_path (str): Path to energy dataset.
        weather_path (str): Path to weather dataset.

    Returns:
        Dict[str, pd.DataFrame]: Dictionary containing energy and weather DataFrames.
    """
    from src.data.ingestion import ingest_data

    logger.info("Starting data ingestion step...")
    data = ingest_data(energy_path, weather_path)
    logger.info(f"Data ingestion completed. Energy shape: {data['energy'].shape}, Weather shape: {data['weather'].shape}")

    return data


@step
def preprocess_data_step(data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    ZenML step for data preprocessing.

    Args:
        data (Dict[str, pd.DataFrame]): Raw data from ingestion step.

    Returns:
        pd.DataFrame: Preprocessed and merged DataFrame.
    """
    from src.data.preprocessing import full_preprocessing_pipeline

    logger.info("Starting data preprocessing step...")
    processed_data = full_preprocessing_pipeline(data['energy'], data['weather'])
    logger.info(f"Data preprocessing completed. Shape: {processed_data.shape}")

    return processed_data


@step
def feature_engineering_step(processed_data: pd.DataFrame) -> pd.DataFrame:
    """
    ZenML step for feature engineering.

    Args:
        processed_data (pd.DataFrame): Preprocessed data.

    Returns:
        pd.DataFrame: Data with engineered features.
    """
    from src.features.feature_engineering import full_feature_engineering_pipeline

    logger.info("Starting feature engineering step...")
    feature_data = full_feature_engineering_pipeline(processed_data)
    logger.info(f"Feature engineering completed. Shape: {feature_data.shape}, Features: {len(feature_data.columns)}")

    return feature_data


@enable_mlflow
@step
def train_models_step(feature_data: pd.DataFrame, target_column: str = 'total_load_actual') -> Dict[str, Dict[str, Any]]:
    """
    ZenML step for model training with MLflow experiment tracking.

    Args:
        feature_data (pd.DataFrame): Data with engineered features.
        target_column (str): Target column name.

    Returns:
        Dict[str, Dict[str, Any]]: Training results for all models.
    """
    from src.models.train import train_multiple_models

    logger.info("Starting model training step...")

    # Start MLflow experiment
    mlflow.set_experiment("energy_demand_forecasting")
    with mlflow.start_run(run_name="model_training"):
        # Log dataset info
        mlflow.log_param("dataset_shape", feature_data.shape)
        mlflow.log_param("target_column", target_column)
        mlflow.log_param("feature_count", len(feature_data.columns))

        results = train_multiple_models(feature_data, target_column=target_column)

        # Log metrics and models to MLflow
        for model_name, result in results.items():
            with mlflow.start_run(run_name=f"{model_name}_training", nested=True):
                # Log hyperparameters
                if 'params' in result:
                    for param_name, param_value in result['params'].items():
                        mlflow.log_param(f"{model_name}_{param_name}", param_value)

                # Log metrics
                metrics = result['metrics']
                for metric_name, metric_value in metrics.items():
                    mlflow.log_metric(f"{model_name}_{metric_name}", metric_value)

                logger.info(f"{model_name.upper()} - MAE: {metrics['mae']:.4f}, R2: {metrics['r2']:.4f}")

                # Register model in MLflow Model Registry
                model_registry_name = f"energy_demand_{model_name}_model"
                try:
                    if model_name == 'arima':
                        # For ARIMA, log parameters as model metadata
                        mlflow.log_dict(result['params'], f"{model_name}_params.json")
                        model_uri = f"runs:/{mlflow.active_run().info.run_id}/{model_name}_params.json"
                        registered_model = mlflow.register_model(model_uri, model_registry_name)

                    elif model_name == 'prophet':
                        # Prophet models are saved as artifacts
                        model_uri = f"runs:/{mlflow.active_run().info.run_id}/{model_name}_prophet.pkl"
                        registered_model = mlflow.register_model(model_uri, model_registry_name)

                    elif model_name == 'lstm':
                        # Register the LSTM model
                        model_uri = f"runs:/{mlflow.active_run().info.run_id}/{model_name}_lstm.h5"
                        registered_model = mlflow.register_model(model_uri, model_registry_name)

                    logger.info(f"Model {model_registry_name} registered with version {registered_model.version}")

                except Exception as e:
                    logger.warning(f"Could not register {model_name} model: {e}")

        # Log overall best model
        best_model = min(results.keys(), key=lambda x: results[x]['metrics']['mae'])
        mlflow.log_param("best_model", best_model)
        mlflow.log_metric("best_model_mae", results[best_model]['metrics']['mae'])

    logger.info("Model training completed with MLflow tracking.")
    return results


@step
def evaluate_models_step(training_results: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    """
    ZenML step for model evaluation and selection.

    Args:
        training_results (Dict[str, Dict[str, Any]]): Results from training step.

    Returns:
        Dict[str, Any]: Evaluation summary and best model info.
    """
    logger.info("Starting model evaluation step...")

    # Compare models based on MAE
    model_comparison = {}
    for model_name, result in training_results.items():
        model_comparison[model_name] = result['metrics']['mae']

    best_model = min(model_comparison, key=model_comparison.get)
    best_mae = model_comparison[best_model]

    evaluation_summary = {
        'model_comparison': model_comparison,
        'best_model': best_model,
        'best_mae': best_mae,
        'all_results': training_results
    }

    logger.info(f"Model evaluation completed. Best model: {best_model} with MAE: {best_mae:.4f}")

    return evaluation_summary


@pipeline
def energy_demand_training_pipeline(energy_path: str, weather_path: str, target_column: str = 'total_load_actual'):
    """
    Complete ZenML pipeline for energy demand forecasting training.

    Args:
        energy_path (str): Path to energy dataset.
        weather_path (str): Path to weather dataset.
        target_column (str): Target column for forecasting.
    """
    # Data pipeline
    raw_data = ingest_data_step(energy_path=energy_path, weather_path=weather_path)
    processed_data = preprocess_data_step(data=raw_data)
    feature_data = feature_engineering_step(processed_data=processed_data)

    # Model pipeline
    training_results = train_models_step(feature_data=feature_data, target_column=target_column)
    evaluation = evaluate_models_step(training_results=training_results)


if __name__ == "__main__":
    # Example usage
    import argparse
    from pathlib import Path

    parser = argparse.ArgumentParser(description='Run ZenML training pipeline')
    parser.add_argument('--energy-path', type=str, default='data/raw/energy_dataset.csv',
                       help='Path to energy dataset')
    parser.add_argument('--weather-path', type=str, default='data/raw/weather_features.csv',
                       help='Path to weather dataset')
    parser.add_argument('--target-column', type=str, default='total_load_actual',
                       help='Target column for forecasting')

    args = parser.parse_args()

    # Ensure data paths exist
    energy_path = Path(args.energy_path)
    weather_path = Path(args.weather_path)

    if not energy_path.exists():
        raise FileNotFoundError(f"Energy dataset not found at {energy_path}")
    if not weather_path.exists():
        raise FileNotFoundError(f"Weather dataset not found at {weather_path}")

    # Run pipeline
    logger.info("Starting ZenML training pipeline...")
    energy_demand_training_pipeline(
        energy_path=str(energy_path),
        weather_path=str(weather_path),
        target_column=args.target_column
    )
    logger.info("ZenML training pipeline completed.")

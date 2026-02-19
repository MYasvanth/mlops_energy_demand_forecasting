#!/usr/bin/env python3
"""
Single Model Training Script

Train individual models (ARIMA, Prophet, LSTM, XGBoost, LightGBM) for energy demand forecasting.

Usage:
    python scripts/training/train_single_model.py --model xgboost
    python scripts/training/train_single_model.py --model lightgbm --no-tune
    python scripts/training/train_single_model.py --model arima --target total_load_actual

NOTE: This script uses consistent feature engineering across all model types:
- Traditional models (ARIMA, Prophet, LSTM): Use full_feature_engineering_pipeline()
- GBM models (XGBoost, LightGBM): Use GBMFeatureEngineer for optimal performance
"""

import argparse
import logging
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.models.train import TimeSeriesTrainer, train_gbm_models
from src.data.ingestion import ingest_data
from src.data.preprocessing import full_preprocessing_pipeline
from src.features.feature_engineering import full_feature_engineering_pipeline
from src.models.gbm_features import GBMFeatureEngineer
import mlflow
import mlflow.sklearn
import mlflow.pyfunc
import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def train_traditional_model(model_type: str, target_column: str, feature_data: pd.DataFrame,
                           tune_hyperparams: bool = True) -> tuple:
    """
    Train traditional models (ARIMA, Prophet, LSTM) with consistent pipeline.
    
    Args:
        model_type: Type of model ('arima', 'prophet', 'lstm')
        target_column: Target column name
        feature_data: Pre-engineered feature data from full_feature_engineering_pipeline
        tune_hyperparams: Whether to tune hyperparameters
        
    Returns:
        Tuple of (result dict, trainer instance)
    """
    trainer = TimeSeriesTrainer(model_type=model_type, target_column=target_column)
    result = trainer.train(feature_data, tune_hyperparams=tune_hyperparams)
    return result, trainer


def train_gbm_model(model_type: str, target_column: str, processed_data: pd.DataFrame,
                    tune_hyperparams: bool = True) -> dict:
    """
    Train GBM models (XGBoost, LightGBM) with GBMFeatureEngineer.
    
    This ensures GBM models use the same feature engineering as train_gbm_models()
    for consistency across the codebase.
    
    Args:
        model_type: Type of model ('xgboost', 'lightgbm')
        target_column: Target column name
        processed_data: Preprocessed data (before feature engineering)
        tune_hyperparams: Whether to tune hyperparameters
        
    Returns:
        Dictionary with model, metrics, and params
    """
    # Use train_gbm_models for consistent feature engineering
    results = train_gbm_models(
        df=processed_data,
        target_column=target_column,
        models=[model_type],
        n_trials=20 if tune_hyperparams else 5
    )
    
    return results.get(model_type, {'model': None, 'metrics': {}, 'params': {}})


def main():
    parser = argparse.ArgumentParser(description="Train a single energy demand forecasting model")
    
    parser.add_argument('--model', type=str, required=True,
                       choices=['arima', 'prophet', 'lstm', 'xgboost', 'lightgbm'],
                       help='Model type to train')
    parser.add_argument('--target', type=str, default='total_load_actual',
                       help='Target column name')
    parser.add_argument('--no-tune', action='store_true',
                       help='Skip hyperparameter tuning (use defaults)')
    parser.add_argument('--energy-path', type=str, default='data/raw/energy_dataset.csv',
                       help='Path to energy dataset')
    parser.add_argument('--weather-path', type=str, default='data/raw/weather_features.csv',
                       help='Path to weather dataset')
    
    args = parser.parse_args()
    
    # Determine model category
    is_gbm_model = args.model in ['xgboost', 'lightgbm']
    
    try:
        # Set MLflow experiment
        mlflow.set_experiment("Energy Demand Forecasting")
        
        # Load data
        logger.info(f"Loading data from {args.energy_path} and {args.weather_path}")
        raw_data = ingest_data(args.energy_path, args.weather_path)
        
        # Preprocess
        logger.info("Preprocessing data...")
        processed_data = full_preprocessing_pipeline(raw_data['energy'], raw_data['weather'])
        
        # Feature engineering - DIFFERENT PATHS FOR DIFFERENT MODEL TYPES
        if is_gbm_model:
            # GBM models: Use GBMFeatureEngineer for consistency with train_gbm_models()
            logger.info("Using GBMFeatureEngineer for GBM model feature engineering...")
            # Note: GBMFeatureEngineer will be applied inside train_gbm_models()
            feature_data = processed_data
        else:
            # Traditional models: Use full_feature_engineering_pipeline
            logger.info("Engineering features with full_feature_engineering_pipeline...")
            feature_data = full_feature_engineering_pipeline(processed_data)
        
        # Train model with MLflow tracking
        logger.info(f"Training {args.model.upper()} model...")
        
        # End any existing run before starting
        if mlflow.active_run():
            mlflow.end_run()
            
        with mlflow.start_run(run_name=f"{args.model}_training") as run:
            # Log parameters
            mlflow.log_param("model_type", args.model)
            mlflow.log_param("target_column", args.target)
            mlflow.log_param("tune_hyperparams", not args.no_tune)
            mlflow.log_param("feature_engineering", "GBMFeatureEngineer" if is_gbm_model else "full_feature_engineering_pipeline")
            
            # Train based on model type
            if is_gbm_model:
                result = train_gbm_model(
                    model_type=args.model,
                    target_column=args.target,
                    processed_data=processed_data,
                    tune_hyperparams=not args.no_tune
                )
                trainer = None
            else:
                result, trainer = train_traditional_model(
                    model_type=args.model,
                    target_column=args.target,
                    feature_data=feature_data,
                    tune_hyperparams=not args.no_tune
                )
            
            # Log metrics inside the same run
            if result.get('model') is not None:
                metrics = result['metrics']
                mlflow.log_metric("mae", metrics['mae'])
                mlflow.log_metric("rmse", metrics['rmse'])
                mlflow.log_metric("r2", metrics['r2'])
                if 'mse' in metrics:
                    mlflow.log_metric("mse", metrics['mse'])
                
                if 'params' in result and result['params']:
                    for param, value in result['params'].items():
                        mlflow.log_param(f"best_{param}", value)
                
                if 'feature_importance' in result and result['feature_importance']:
                    for feat, imp in sorted(result['feature_importance'].items(), key=lambda x: x[1], reverse=True)[:5]:
                        mlflow.log_metric(f"importance_{feat}", imp)
                
                model_path = f"models/{args.model}_model"
                if trainer:
                    trainer.save_model(model_path)
                
                try:
                    model_name = f"energy_demand_{args.model}_model"
                    if args.model in ['xgboost', 'lightgbm']:
                        mlflow.sklearn.log_model(
                            sk_model=result['model'].model if hasattr(result['model'], 'model') else result['model'],
                            artifact_path=f"{args.model}_model",
                            registered_model_name=model_name
                        )
                    elif args.model == 'arima':
                        class ARIMAWrapper(mlflow.pyfunc.PythonModel):
                            def load_context(self, context):
                                import joblib
                                self.model = joblib.load(context.artifacts['model'])
                            def predict(self, context, model_input):
                                return self.model.forecast(steps=len(model_input) if hasattr(model_input, '__len__') else 1)
                        mlflow.pyfunc.log_model(artifact_path="arima_model", python_model=ARIMAWrapper(),
                                              artifacts={'model': model_path + '_fitted.pkl'}, registered_model_name=model_name)
                    elif args.model == 'prophet':
                        class ProphetWrapper(mlflow.pyfunc.PythonModel):
                            def load_context(self, context):
                                import joblib
                                self.model = joblib.load(context.artifacts['model'])
                            def predict(self, context, model_input):
                                return self.model.predict(model_input)['yhat'].values
                        mlflow.pyfunc.log_model(artifact_path="prophet_model", python_model=ProphetWrapper(),
                                              artifacts={'model': model_path + '_prophet.pkl'}, registered_model_name=model_name)
                    elif args.model == 'lstm':
                        class LSTMWrapper(mlflow.pyfunc.PythonModel):
                            def load_context(self, context):
                                import tensorflow as tf
                                import joblib
                                self.model = tf.keras.models.load_model(context.artifacts['model'])
                                self.scaler = joblib.load(context.artifacts['scaler'])
                            def predict(self, context, model_input):
                                scaled = self.scaler.transform(model_input.reshape(-1, 1))
                                return self.scaler.inverse_transform(self.model.predict(scaled.reshape((scaled.shape[0], scaled.shape[1], 1)))).flatten()
                        mlflow.pyfunc.log_model(artifact_path="lstm_model", python_model=LSTMWrapper(),
                                              artifacts={'model': model_path + '_lstm.h5', 'scaler': model_path + '_scaler.pkl'},
                                              registered_model_name=model_name)
                    
                    client = mlflow.tracking.MlflowClient()
                    model_versions = client.search_model_versions(f"name='{model_name}'")
                    if model_versions:
                        client.transition_model_version_stage(name=model_name, version=model_versions[0].version, stage="Staging")
                except Exception as e:
                    logger.warning(f"Failed to register model: {e}")
            else:
                mlflow.log_param("status", "failed")
                if 'error' in result:
                    mlflow.log_param("error", str(result['error'])[:250])
        
        # Print results
        print("\n" + "="*60)
        print(f"{args.model.upper()} TRAINING RESULTS")
        print("="*60)
        
        if result.get('model') is not None:
            metrics = result['metrics']
            print(f"MAE:  {metrics['mae']:.4f}")
            print(f"RMSE: {metrics['rmse']:.4f}")
            print(f"R2:   {metrics['r2']:.4f}")
            if 'params' in result and result['params']:
                print(f"\nBest Parameters: {result['params']}")
            if 'feature_importance' in result and result['feature_importance']:
                print("\nTop 5 Important Features:")
                for feat, imp in sorted(result['feature_importance'].items(), key=lambda x: x[1], reverse=True)[:5]:
                    print(f"  {feat}: {imp:.4f}")
            print(f"\nModel saved and registered in MLflow")
        else:
            print("Training failed!")
            if 'error' in result:
                print(f"Error: {result['error']}")
        
        print("="*60)
        
    except Exception as e:
        logger.error(f"Training failed: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()

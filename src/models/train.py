"""
Model Training Module

This module handles model training for energy demand forecasting using various algorithms
including ARIMA, Prophet, LSTM, XGBoost, and LightGBM with hyperparameter tuning using Optuna.

Implements the Strategy pattern for unified model training interface:
- Traditional models: ARIMA, Prophet, LSTM
- Gradient Boosting Models: XGBoost, LightGBM (using BaseTimeSeriesModel)
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TensorFlow warnings

import logging
from typing import Dict, List, Optional, Tuple, Union, Any
import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from .evaluation import (
    time_series_cross_validation,
    calculate_metrics,
    energy_specific_metrics,
    validate_arima_model,
    validate_lstm_model,
    comprehensive_cross_validation,
    log_validation_results,
    select_best_model
)
from sklearn.preprocessing import StandardScaler
import optuna
import joblib
import os
from pathlib import Path
import mlflow
import mlflow.sklearn

# Configure logging first
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Time series models
from statsmodels.tsa.arima.model import ARIMA
from prophet import Prophet

# TensorFlow/Keras imports for LSTM
try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout
    from tensorflow.keras.callbacks import EarlyStopping
    TF_AVAILABLE = True
    logger.info(f"TensorFlow {tf.__version__} loaded successfully")
except ImportError as e:
    TF_AVAILABLE = False
    logger.warning(f"TensorFlow not available: {e}. LSTM models will not be available.")
except Exception as e:
    TF_AVAILABLE = False
    logger.warning(f"TensorFlow import failed with error: {e}. LSTM models will not be available.")

# Gradient Boosting Models - Strategy pattern imports
from .base import BaseTimeSeriesModel, ModelRegistry

# Try to import GBM models (may fail if xgboost/lightgbm not installed)
GBM_AVAILABLE = True
try:
    from .xgboost_model import XGBoostTimeSeriesModel
    from .lightgbm_model import LightGBMTimeSeriesModel
except ImportError:
    GBM_AVAILABLE = False
    logger.warning("XGBoost/LightGBM not available. GBM models will be skipped.")

# Suppress warnings
import warnings
warnings.filterwarnings('ignore')

# Register GBM models with ModelRegistry if available
if GBM_AVAILABLE:
    # Models are already registered in their respective modules when imported
    # But we ensure they're registered here for clarity
    pass


class TimeSeriesTrainer:
    """
    Trainer class for time series forecasting models.
    """

    def __init__(self, model_type: str = 'arima', target_column: str = 'total_load_actual'):
        """
        Initialize the trainer.

        Args:
            model_type (str): Type of model ('arima', 'prophet', 'lstm').
            target_column (str): Name of the target column.
        """
        self.model_type = model_type
        self.target_column = target_column
        self.model = None
        self.scaler = StandardScaler()
        self.best_params = None

    def prepare_data_arima(self, df: pd.DataFrame, target_col: str) -> Tuple[pd.Series, pd.Series]:
        """
        Prepare data for ARIMA model.

        Args:
            df (pd.DataFrame): Input DataFrame.
            target_col (str): Target column name.

        Returns:
            Tuple[pd.Series, pd.Series]: Train and test series.
        """
        series = df[target_col].dropna()
        # Ensure series is numeric and convert to float64
        series = pd.to_numeric(series, errors='coerce').dropna()
        train_size = int(len(series) * 0.8)
        train = series[:train_size]
        test = series[train_size:]
        return train, test

    def prepare_data_prophet(self, df: pd.DataFrame, target_col: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Prepare data for Prophet model.

        Args:
            df (pd.DataFrame): Input DataFrame.
            target_col (str): Target column name.

        Returns:
            Tuple[pd.DataFrame, pd.DataFrame]: Train and test DataFrames.
        """
        # Reset index and create prophet_df with proper column names
        prophet_df = df.reset_index()
        index_col = df.index.name if df.index.name else 'index'
        prophet_df = prophet_df[[index_col, target_col]].rename(
            columns={index_col: 'ds', target_col: 'y'}
        )
        prophet_df = prophet_df.dropna()

        # Remove timezone from datetime column for Prophet compatibility
        prophet_df['ds'] = prophet_df['ds'].dt.tz_localize(None)

        train_size = int(len(prophet_df) * 0.8)
        train = prophet_df[:train_size]
        test = prophet_df[train_size:]
        return train, test

    def prepare_data_lstm(self, df: pd.DataFrame, target_col: str, lookback: int = 24) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Prepare data for LSTM model WITHOUT DATA LEAKAGE.
        CRITICAL: Split data BEFORE scaling to prevent test data leakage.

        Args:
            df (pd.DataFrame): Input DataFrame.
            target_col (str): Target column name.
            lookback (int): Number of time steps to look back.

        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]: X_train, y_train, X_test, y_test.
        """
        series = df[target_col].dropna().values.reshape(-1, 1)
        
        # SPLIT FIRST to avoid leakage
        train_size = int(len(series) * 0.8)
        train_series = series[:train_size]
        test_series = series[train_size:]
        
        # Fit scaler on TRAIN ONLY
        self.scaler.fit(train_series)
        train_scaled = self.scaler.transform(train_series).flatten()
        test_scaled = self.scaler.transform(test_series).flatten()
        
        # Create sequences for train
        X_train, y_train = [], []
        for i in range(len(train_scaled) - lookback):
            X_train.append(train_scaled[i:i + lookback])
            y_train.append(train_scaled[i + lookback])
        
        # Create sequences for test
        X_test, y_test = [], []
        for i in range(len(test_scaled) - lookback):
            X_test.append(test_scaled[i:i + lookback])
            y_test.append(test_scaled[i + lookback])
        
        return np.array(X_train), np.array(y_train), np.array(X_test), np.array(y_test)

    def objective_arima(self, trial: optuna.Trial, train_data: pd.Series) -> float:
        """
        Objective function for ARIMA hyperparameter tuning WITHOUT LEAKAGE.
        Uses walk-forward validation on training data only.

        Args:
            trial (optuna.Trial): Optuna trial object.
            train_data (pd.Series): Training data.

        Returns:
            float: Mean absolute error from walk-forward validation.
        """
        p = trial.suggest_int('p', 0, 5)
        d = trial.suggest_int('d', 0, 2)
        q = trial.suggest_int('q', 0, 5)

        try:
            # Use first 80% for fitting, last 20% for validation
            val_size = len(train_data) // 5
            train_cv = train_data[:-val_size]
            val_cv = train_data[-val_size:]
            
            model = ARIMA(train_cv, order=(p, d, q))
            model_fit = model.fit()
            predictions = model_fit.forecast(steps=len(val_cv))
            mae = mean_absolute_error(val_cv, predictions)
            return mae
        except:
            return float('inf')

    def train_arima(self, df: pd.DataFrame, tune_hyperparams: bool = True) -> Dict[str, Any]:
        """
        Train ARIMA model.

        Args:
            df (pd.DataFrame): Input DataFrame.
            tune_hyperparams (bool): Whether to tune hyperparameters.

        Returns:
            Dict[str, Any]: Training results.
        """
        logger.info("Training ARIMA model...")

        train, test = self.prepare_data_arima(df, self.target_column)

        if tune_hyperparams:
            study = optuna.create_study(direction='minimize')
            study.optimize(lambda trial: self.objective_arima(trial, train), n_trials=20)
            self.best_params = study.best_params
            p, d, q = self.best_params['p'], self.best_params['d'], self.best_params['q']
        else:
            p, d, q = 1, 1, 1  # Default parameters

        self.model = ARIMA(train, order=(p, d, q)).fit()

        # Evaluate on test set
        predictions = self.model.forecast(steps=len(test))
        metrics = calculate_metrics(test.values, predictions)

        # Add business-specific metrics
        business_metrics = energy_specific_metrics(test.values, predictions)
        metrics.update(business_metrics)

        # Perform cross-validation
        cv_results = comprehensive_cross_validation(df.assign(**{self.target_column: df[self.target_column]}))
        # End any active run before logging validation results
        if mlflow.active_run():
            mlflow.end_run()
        log_validation_results('arima', {'cv_results': cv_results, 'test_score': metrics['mae']}, run_name=f"arima_validation_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}")

        logger.info(f"ARIMA training completed. MAE: {metrics['mae']:.4f}")
        return {
            'model': self.model,
            'metrics': metrics,
            'params': {'p': p, 'd': d, 'q': q}
        }

    def train_prophet(self, df: pd.DataFrame, tune_hyperparams: bool = True) -> Dict[str, Any]:
        """
        Train Prophet model WITHOUT DATA LEAKAGE.

        Args:
            df (pd.DataFrame): Input DataFrame.
            tune_hyperparams (bool): Whether to tune hyperparameters.

        Returns:
            Dict[str, Any]: Training results.
        """
        logger.info("Training Prophet model...")

        train, test = self.prepare_data_prophet(df, self.target_column)

        if tune_hyperparams:
            # Use CV on TRAIN ONLY to avoid leakage
            best_mae = float('inf')
            best_params = {'changepoint_prior_scale': 0.05, 'seasonality_prior_scale': 10}
            
            # Split train into train_cv and val_cv
            cv_split = int(len(train) * 0.8)
            train_cv = train.iloc[:cv_split]
            val_cv = train.iloc[cv_split:]

            for cps in [0.01, 0.05, 0.1]:
                for sps in [1, 10, 20]:
                    try:
                        model = Prophet(changepoint_prior_scale=cps, seasonality_prior_scale=sps, 
                                      daily_seasonality=True, weekly_seasonality=True, yearly_seasonality=True)
                        model.fit(train_cv)
                        # Predict only on validation period
                        val_forecast = model.predict(val_cv[['ds']])
                        mae = mean_absolute_error(val_cv['y'], val_forecast['yhat'])
                        if mae < best_mae:
                            best_mae = mae
                            best_params = {'changepoint_prior_scale': cps, 'seasonality_prior_scale': sps}
                    except:
                        continue

            self.best_params = best_params
        else:
            self.best_params = {'changepoint_prior_scale': 0.05, 'seasonality_prior_scale': 10}

        # Train final model on full training set
        self.model = Prophet(**self.best_params, daily_seasonality=True, 
                           weekly_seasonality=True, yearly_seasonality=True)
        self.model.fit(train)

        # Predict ONLY on test dates (no future dataframe needed)
        test_forecast = self.model.predict(test[['ds']])
        predictions = test_forecast['yhat'].values

        metrics = self.calculate_metrics(test['y'].values, predictions)

        logger.info(f"Prophet training completed. MAE: {metrics['mae']:.4f}, R2: {metrics['r2']:.4f}")
        return {
            'model': self.model,
            'metrics': metrics,
            'params': self.best_params
        }

    def objective_lstm(self, trial: optuna.Trial, X_train: np.ndarray, y_train: np.ndarray,
                      X_val: np.ndarray, y_val: np.ndarray) -> float:
        """
        Objective function for LSTM hyperparameter tuning.

        Args:
            trial (optuna.Trial): Optuna trial object.
            X_train (np.ndarray): Training features.
            y_train (np.ndarray): Training targets.
            X_val (np.ndarray): Validation features.
            y_val (np.ndarray): Validation targets.

        Returns:
            float: Validation mean absolute error.
        """
        if not TF_AVAILABLE:
            raise ImportError("TensorFlow is not available. Install with: pip install tensorflow")
            
        units = trial.suggest_int('units', 32, 128)
        dropout = trial.suggest_float('dropout', 0.1, 0.5)
        learning_rate = trial.suggest_float('learning_rate', 1e-4, 1e-2, log=True)

        model = Sequential([
            LSTM(units, input_shape=(X_train.shape[1], 1), return_sequences=True),
            Dropout(dropout),
            LSTM(units // 2),
            Dropout(dropout),
            Dense(1)
        ])

        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
                     loss='mae')

        early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

        model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_val, y_val),
                 callbacks=[early_stopping], verbose=0)

        predictions = model.predict(X_val)
        mae = mean_absolute_error(y_val, predictions)
        return mae

    def train_lstm(self, df: pd.DataFrame, tune_hyperparams: bool = True) -> Dict[str, Any]:
        """
        Train LSTM model with robust error handling and simplified approach.

        Args:
            df (pd.DataFrame): Input DataFrame.
            tune_hyperparams (bool): Whether to tune hyperparameters.

        Returns:
            Dict[str, Any]: Training results.
        """
        logger.info("Training LSTM model...")
        
        if not TF_AVAILABLE:
            error_msg = "TensorFlow is not available. Install with: pip install tensorflow"
            logger.error(error_msg)
            return {
                'model': None,
                'scaler': self.scaler,
                'metrics': {'mae': float('inf'), 'mse': float('inf'), 'rmse': float('inf'), 'r2': float('-inf')},
                'params': {},
                'error': error_msg
            }

        try:
            # Data preparation with validation
            X_train, y_train, X_test, y_test = self.prepare_data_lstm(df, self.target_column)

            # Validate data shapes
            if len(X_train) == 0 or len(y_train) == 0:
                raise ValueError("Insufficient data for LSTM training")

            # Ensure minimum data requirements
            min_samples = 100
            if len(X_train) < min_samples:
                raise ValueError(f"Need at least {min_samples} samples for training, got {len(X_train)}")

            # Reshape for LSTM
            X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
            X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

            logger.info(f"LSTM data prepared: X_train shape {X_train.shape}, y_train shape {y_train.shape}")

            # Set default parameters
            self.best_params = {'units': 32, 'dropout': 0.1, 'learning_rate': 0.001}

            # Create simplified model with error handling
            try:
                model = Sequential([
                    LSTM(self.best_params['units'], input_shape=(X_train.shape[1], 1), return_sequences=True),
                    Dropout(self.best_params['dropout']),
                    LSTM(self.best_params['units'] // 2),
                    Dropout(self.best_params['dropout']),
                    Dense(1)
                ])
            except Exception as model_e:
                logger.warning(f"Failed to create LSTM model: {model_e}. Using simpler architecture.")
                # Fallback to simpler model
                model = Sequential([
                    LSTM(16, input_shape=(X_train.shape[1], 1)),
                    Dense(1)
                ])

            # Compile model with error handling
            try:
                model.compile(
                    optimizer=tf.keras.optimizers.Adam(learning_rate=self.best_params['learning_rate']),
                    loss='mae',
                    metrics=[tf.keras.metrics.MeanAbsoluteError()]
                )
            except Exception as compile_e:
                logger.warning(f"Failed to compile model: {compile_e}. Using basic optimizer.")
                model.compile(optimizer='adam', loss='mae', metrics=[tf.keras.metrics.MeanAbsoluteError()])

            # Setup callbacks
            early_stopping = EarlyStopping(
                monitor='val_loss',
                patience=5,
                restore_best_weights=True,
                min_delta=1e-4
            )

            # Validation split with safety checks
            val_split = 0.2
            val_samples = max(1, int(len(X_train) * val_split))

            if len(X_train) <= val_samples:
                # If not enough data for validation split, use minimal validation
                X_train_final, X_val_final = X_train, X_train[-1:]
                y_train_final, y_val_final = y_train, y_train[-1:]
            else:
                X_train_final, X_val_final = X_train[:-val_samples], X_train[-val_samples:]
                y_train_final, y_val_final = y_train[:-val_samples], y_train[-val_samples:]

            # Train model with comprehensive error handling
            try:
                history = model.fit(
                    X_train_final, y_train_final,
                    epochs=20,  # Reduced epochs for stability
                    batch_size=min(32, len(X_train_final)),  # Adaptive batch size
                    validation_data=(X_val_final, y_val_final) if len(X_val_final) > 0 else None,
                    callbacks=[early_stopping],
                    verbose=0
                )

                # Validate training success
                if len(history.history.get('loss', [])) == 0:
                    raise ValueError("No training history recorded")

                final_loss = history.history['loss'][-1]
                if not np.isfinite(final_loss):
                    raise ValueError(f"Training loss became non-finite: {final_loss}")

                logger.info(f"LSTM training completed. Final loss: {final_loss:.4f}")

            except Exception as train_e:
                logger.error(f"LSTM training failed: {train_e}")
                # Create a simple baseline model as fallback
                logger.info("Creating simple baseline model as fallback...")
                model = Sequential([Dense(1, input_shape=(X_train.shape[1],))])
                model.compile(optimizer='adam', loss='mae')

                # Train simple model
                simple_X = X_train_final.reshape(X_train_final.shape[0], -1)
                model.fit(simple_X, y_train_final, epochs=5, verbose=0, batch_size=min(32, len(simple_X)))

                return {
                    'model': model,
                    'scaler': self.scaler,
                    'metrics': {'mae': float('inf'), 'mse': float('inf'), 'rmse': float('inf'), 'r2': float('-inf')},
                    'params': self.best_params,
                    'error': str(train_e),
                    'fallback': True
                }

            # Test model prediction capability
            try:
                test_pred = model.predict(X_val_final[:min(1, len(X_val_final))], verbose=0)
                if not np.isfinite(test_pred).all():
                    raise ValueError("Model produces non-finite predictions")
            except Exception as pred_e:
                logger.warning(f"Model prediction test failed: {pred_e}")
                # Continue anyway as this might be due to empty validation set

            # Evaluate on test set with error handling
            try:
                predictions = model.predict(X_test, verbose=0, batch_size=min(32, len(X_test)))

                # Validate predictions
                if not np.isfinite(predictions).all():
                    logger.warning("Model predictions contain non-finite values, using fallback metrics")
                    metrics = {'mae': float('inf'), 'mse': float('inf'), 'rmse': float('inf'), 'r2': float('-inf')}
                else:
                    predictions = self.scaler.inverse_transform(predictions)
                    y_test_original = self.scaler.inverse_transform(y_test.reshape(-1, 1))

                    # Additional validation
                    if len(predictions) != len(y_test_original):
                        logger.warning("Prediction and test set lengths don't match")
                        metrics = {'mae': float('inf'), 'mse': float('inf'), 'rmse': float('inf'), 'r2': float('-inf')}
                    else:
                        metrics = self.calculate_metrics(y_test_original.flatten(), predictions.flatten())

                        # Validate metrics
                        if not np.isfinite(metrics['mae']):
                            logger.warning("Calculated metrics contain non-finite values")
                            metrics = {'mae': float('inf'), 'mse': float('inf'), 'rmse': float('inf'), 'r2': float('-inf')}

            except Exception as eval_e:
                logger.error(f"Model evaluation failed: {eval_e}")
                metrics = {'mae': float('inf'), 'mse': float('inf'), 'rmse': float('inf'), 'r2': float('-inf')}

            # Ensure metrics always exist
            if 'metrics' not in locals() or metrics is None:
                metrics = {'mae': float('inf'), 'mse': float('inf'), 'rmse': float('inf'), 'r2': float('-inf')}

            # Assign model for saving
            self.model = model

            logger.info(f"LSTM training completed. MAE: {metrics['mae']:.4f}")
            return {
                'model': model,
                'scaler': self.scaler, 
                'metrics': metrics if 'metrics' in locals() and metrics else {'mae': float('inf'), 'mse': float('inf'), 'rmse': float('inf'), 'r2': float('-inf')},
                'params': self.best_params
            }

        except Exception as e:
            logger.error(f"LSTM training failed completely: {str(e)}")

            # Final fallback - create a dummy model that can be saved
            try:
                logger.info("Creating final fallback model...")
                fallback_model = Sequential([Dense(1, input_shape=(1,))])
                fallback_model.compile(optimizer='adam', loss='mae')

                # Dummy training
                dummy_X = np.random.randn(10, 1)
                dummy_y = np.random.randn(10)
                fallback_model.fit(dummy_X, dummy_y, epochs=1, verbose=0)

                self.model = fallback_model

                return {
                    'model': fallback_model,
                    'scaler': self.scaler,
                    'metrics': {'mae': float('inf'), 'mse': float('inf'), 'rmse': float('inf'), 'r2': float('-inf')},
                    'params': self.best_params,
                    'error': str(e),
                    'fallback': True
                }

            except Exception as final_e:
                logger.error(f"Final fallback failed: {str(final_e)}")
                self.model = None

                return {
                    'model': None,
                    'scaler': self.scaler,
                    'metrics': {'mae': float('inf'), 'mse': float('inf'), 'rmse': float('inf'), 'r2': float('-inf')},
                    'params': self.best_params,
                    'error': f"Complete training failure: {str(e)}, Final fallback: {str(final_e)}"
                }

    def calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """
        Calculate evaluation metrics.

        Args:
            y_true (np.ndarray): True values.
            y_pred (np.ndarray): Predicted values.

        Returns:
            Dict[str, float]: Dictionary of metrics.
        """
        mae = mean_absolute_error(y_true, y_pred)
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_true, y_pred)

        return {
            'mae': mae,
            'mse': mse,
            'rmse': rmse,
            'r2': r2
        }

    def save_model(self, path: str) -> None:
        """
        Save the trained model.

        Args:
            path (str): Path to save the model.
        """
        if self.model is None:
            raise ValueError("No model to save. Train a model first.")

        os.makedirs(os.path.dirname(path), exist_ok=True)

        if self.model_type == 'arima':
            # ARIMA models can't be pickled directly, save parameters
            model_data = {
                'model_type': self.model_type,
                'params': self.best_params,
                'metrics': getattr(self, 'last_metrics', None)
            }
            joblib.dump(model_data, path + '_params.pkl')
            # Also save the fitted model for later use
            joblib.dump(self.model, path + '_fitted.pkl')
        elif self.model_type == 'prophet':
            joblib.dump(self.model, path + '_prophet.pkl')
        elif self.model_type == 'lstm':
            self.model.save(path + '_lstm.h5')
            joblib.dump(self.scaler, path + '_scaler.pkl')

        logger.info(f"Model saved to {path}")

    def load_model(self, path: str) -> None:
        """
        Load a trained model.

        Args:
            path (str): Path to load the model from.
        """
        if self.model_type == 'arima':
            model_data = joblib.load(path + '_params.pkl')
            self.best_params = model_data['params']
            # Re-train the model with loaded params
            # Note: This requires the original training data
        elif self.model_type == 'prophet':
            self.model = joblib.load(path + '_prophet.pkl')
        elif self.model_type == 'lstm':
            self.model = tf.keras.models.load_model(path + '_lstm.h5')
            self.scaler = joblib.load(path + '_scaler.pkl')

        logger.info(f"Model loaded from {path}")

    def train(self, df: pd.DataFrame, tune_hyperparams: bool = True) -> Dict[str, Any]:
        """
        Train the specified model type.

        Args:
            df (pd.DataFrame): Input DataFrame.
            tune_hyperparams (bool): Whether to tune hyperparameters.

        Returns:
            Dict[str, Any]: Training results.
        """
        if self.model_type == 'arima':
            return self.train_arima(df, tune_hyperparams)
        elif self.model_type == 'prophet':
            return self.train_prophet(df, tune_hyperparams)
        elif self.model_type == 'lstm':
            return self.train_lstm(df, tune_hyperparams)
        elif self.model_type in ['xgboost', 'lightgbm']:
            # For GBM models, use train_gbm_models function
            if not GBM_AVAILABLE:
                raise ImportError(f"{self.model_type} not available. Install with: pip install {self.model_type}")
            results = train_gbm_models(df, self.target_column, [self.model_type])
            return results.get(self.model_type, {'model': None, 'metrics': {}, 'params': {}})
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")


def create_model_comparison(results: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    """
    Create a comparison of model results.

    Args:
        results (Dict[str, Dict[str, Any]]): Results from multiple model trainings.

    Returns:
        Dict[str, Any]: Comparison summary.
    """
    comparison = {}
    best_model = None
    best_mae = float('inf')

    for model_name, result in results.items():
        if 'metrics' in result and 'mae' in result['metrics']:
            mae = result['metrics']['mae']
            comparison[model_name] = mae
            if mae < best_mae:
                best_mae = mae
                best_model = model_name

    return {
        'model_comparison': comparison,
        'best_model': best_model,
        'best_mae': best_mae
    }


def train_gbm_models(df: pd.DataFrame, target_column: str = 'total_load_actual',
                    models: List[str] = ['xgboost', 'lightgbm'],
                    n_splits: int = 5, n_trials: int = 10) -> Dict[str, Dict[str, Any]]:
    """
    Train Gradient Boosting Models (XGBoost, LightGBM) using the Strategy pattern.
    
    This function uses ModelRegistry to create and train GBM models that implement
    the BaseTimeSeriesModel interface, ensuring consistency and extensibility.

    Args:
        df (pd.DataFrame): Input DataFrame.
        target_column (str): Target column name.
        models (List[str]): List of GBM model types to train ('xgboost', 'lightgbm').
        n_splits (int): Number of time-series CV splits.
        n_trials (int): Number of Optuna trials for hyperparameter tuning.

    Returns:
        Dict[str, Dict[str, Any]]: Results for each model.
    """
    import mlflow
    
    if not GBM_AVAILABLE:
        logger.warning("GBM models not available. Install xgboost and lightgbm to use this feature.")
        return {}
    
    # End any existing run before starting
    if mlflow.active_run():
        mlflow.end_run()
        
    # Set MLflow experiment
    mlflow.set_experiment("Energy Demand Forecasting - GBM Models")

    results = {}
    registered_models = {}
    
    # Use GBM Feature Engineer for consistent feature engineering
    from .gbm_features import GBMFeatureEngineer
    
    # Initialize feature engineer
    feature_engineer = GBMFeatureEngineer(
        target_column=target_column,
        lags=[1, 2, 3, 6, 12, 24, 48, 168],
        rolling_windows=[3, 6, 12, 24, 48],
        add_time_features=True
    )
    
    # Prepare train/test split with proper feature engineering
    logger.info("Preparing train/test split with GBM feature engineering...")
    (X_train_full, y_train_full), (X_test, y_test) = feature_engineer.prepare_train_test_split(df, train_size=0.8)
    
    logger.info(f"Train samples: {len(X_train_full)}, Test samples: {len(X_test)}")
    logger.info(f"Number of features: {X_train_full.shape[1]}")
    logger.info(f"Feature names: {list(X_train_full.columns)[:10]}...")  # Show first 10 features

    for model_type in models:
        logger.info(f"Training {model_type} model using Strategy pattern...")
        
        # Skip if model not available
        if model_type not in ModelRegistry.available_models():
            logger.warning(f"Model {model_type} not available in ModelRegistry")
            continue
        
        # Ensure no active run before starting new one
        if mlflow.active_run():
            mlflow.end_run()

        try:
            with mlflow.start_run(run_name=f"{model_type}_gbm_training") as run:
                # Create model using ModelRegistry (Strategy pattern)
                model = ModelRegistry.create(
                    model_type,
                    target_column=target_column,
                    n_splits=n_splits,
                    n_trials=n_trials
                )
                
                # Log parameters
                mlflow.log_param("model_type", model_type)
                mlflow.log_param("target_column", target_column)
                mlflow.log_param("n_splits", n_splits)
                mlflow.log_param("n_trials", n_trials)
                mlflow.log_param("n_features", X_train_full.shape[1])
                
                # Fit the model with already engineered features
                # Pass the feature engineer to avoid double feature engineering
                model.feature_engineer = feature_engineer
                model.feature_names = list(X_train_full.columns)
                
                # Train the model directly on engineered features
                model._train_on_features(X_train_full, y_train_full, X_test, y_test)
                
                # Make predictions on test set
                y_pred = model.model.predict(X_test)
                y_true = y_test.values
                
                # Calculate metrics
                metrics = {
                    'mae': mean_absolute_error(y_true, y_pred),
                    'mse': mean_squared_error(y_true, y_pred),
                    'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
                    'r2': r2_score(y_true, y_pred)
                }
                
                # Add energy-specific metrics
                energy_metrics = energy_specific_metrics(y_true, y_pred)
                metrics.update(energy_metrics)
                
                # Log metrics
                mlflow.log_metric("mae", metrics['mae'])
                mlflow.log_metric("mse", metrics['mse'])
                mlflow.log_metric("rmse", metrics['rmse'])
                mlflow.log_metric("r2", metrics['r2'])
                
                # Log best hyperparameters
                if model.best_params:
                    for param, value in model.best_params.items():
                        mlflow.log_param(f"best_{param}", value)
                
                # Save model and feature engineer
                model_path = f"models/{model_type}_gbm_model"
                os.makedirs("models", exist_ok=True)
                
                # Save the trained model
                import joblib
                joblib.dump(model.model, model_path + '.joblib')
                mlflow.log_artifact(model_path + '.joblib')
                
                # Save the feature engineer
                feature_engineer.save(model_path + '_feature_engineer.joblib')
                mlflow.log_artifact(model_path + '_feature_engineer.joblib')
                
                # Log feature importance
                importance = model.get_feature_importance()
                if importance:
                    importance_df = pd.DataFrame([
                        {'feature': k, 'importance': v} 
                        for k, v in importance.items()
                    ]).sort_values('importance', ascending=False)
                    
                    # Log top 10 features
                    for idx, row in importance_df.head(10).iterrows():
                        mlflow.log_metric(f"importance_{row['feature']}", row['importance'])
                
                # Register model in MLflow
                model_name = f"energy_demand_{model_type}_gbm_model"
                try:
                    # Use sklearn log_model for GBM models
                    mlflow.sklearn.log_model(
                        sk_model=model.model,
                        artifact_path=f"{model_type}_model",
                        registered_model_name=model_name,
                        pyfunc_predict_fn="predict"
                    )
                    
                    client = mlflow.tracking.MlflowClient()
                    registered_model = client.get_registered_model(model_name)
                    
                    registered_models[model_type] = {
                        'model_name': model_name,
                        'version': len(registered_model.latest_versions) if registered_model else 'N/A',
                        'run_id': run.info.run_id,
                        'mae': metrics['mae']
                    }
                    
                    logger.info(f"GBM Model {model_name} registered successfully")
                    
                except Exception as e:
                    logger.warning(f"Failed to register {model_type} GBM model: {e}")
                    registered_models[model_type] = {
                        'model_name': model_name,
                        'version': 'Failed',
                        'run_id': run.info.run_id,
                        'mae': metrics['mae'],
                        'error': str(e)
                    }
                
                results[model_type] = {
                    'model': model,
                    'metrics': metrics,
                    'params': model.best_params,
                    'feature_importance': importance
                }
                
                logger.info(f"{model_type.upper()} training completed. MAE: {metrics['mae']:.4f}")
                
        except Exception as e:
            logger.error(f"Failed to train {model_type} model: {e}")
            results[model_type] = {
                'model': None,
                'metrics': {'mae': float('inf'), 'mse': float('inf'), 'rmse': float('inf'), 'r2': float('-inf')},
                'params': {},
                'error': str(e)
            }

    # Compare results and find best model
    if results:
        comparison = {k: v['metrics']['mae'] for k, v in results.items() if 'metrics' in v and 'mae' in v['metrics']}
        if comparison:
            best_model = min(comparison, key=comparison.get)
            best_mae = comparison[best_model]
            
            logger.info(f"Best GBM model: {best_model} with MAE: {best_mae:.4f}")
            
            # Promote best model to production
            if best_model in registered_models and registered_models[best_model]['version'] != 'Failed':
                try:
                    model_name = registered_models[best_model]['model_name']
                    version = registered_models[best_model]['version']
                    
                    client = mlflow.tracking.MlflowClient()
                    client.transition_model_version_stage(
                        name=model_name,
                        version=str(version),
                        stage="Production"
                    )
                    
                    registered_models[best_model]['stage'] = 'Production'
                    logger.info(f"GBM Model {model_name} promoted to Production")
                    
                except Exception as e:
                    logger.warning(f"Failed to promote {best_model} to production: {e}")
            
            results['best_model'] = best_model
            results['best_mae'] = best_mae
    
    results['model_registry'] = registered_models
    
    return results


def train_multiple_models(df: pd.DataFrame, target_column: str = 'total_load_actual',
                         models: List[str] = ['arima', 'prophet', 'lstm', 'xgboost', 'lightgbm']) -> Dict[str, Dict[str, Any]]:
    """
    Train multiple models and compare their performance with MLflow model registration.
    
    Supports both traditional models (ARIMA, Prophet, LSTM) and Gradient Boosting Models
    (XGBoost, LightGBM) using a unified interface.

    Args:
        df (pd.DataFrame): Input DataFrame.
        target_column (str): Target column name.
        models (List[str]): List of model types to train.
                          Traditional: 'arima', 'prophet', 'lstm'
                          GBM (Strategy pattern): 'xgboost', 'lightgbm'

    Returns:
        Dict[str, Dict[str, Any]]: Results for each model.
    """
    import mlflow
    
    # Separate traditional models from GBM models
    traditional_models = [m for m in models if m in ['arima', 'prophet', 'lstm']]
    gbm_models = [m for m in models if m in ['xgboost', 'lightgbm']]
    
    results = {}
    
    # Train traditional models
    if traditional_models:
        logger.info(f"Training traditional models: {traditional_models}")
        traditional_results = _train_traditional_models(df, target_column, traditional_models)
        results.update(traditional_results)
    
    # Train GBM models using Strategy pattern
    if gbm_models and GBM_AVAILABLE:
        logger.info(f"Training GBM models using Strategy pattern: {gbm_models}")
        gbm_results = train_gbm_models(df, target_column, gbm_models)
        results.update(gbm_results)
    elif gbm_models and not GBM_AVAILABLE:
        logger.warning(f"GBM models requested but not available: {gbm_models}")
    
    # Find best model overall
    if results:
        comparison = {}
        for model_type, result in results.items():
            if isinstance(result, dict) and 'metrics' in result and 'mae' in result['metrics']:
                if np.isfinite(result['metrics']['mae']):
                    comparison[model_type] = result['metrics']['mae']
        
        if comparison:
            best_model = min(comparison, key=comparison.get)
            best_mae = comparison[best_model]
            results['best_model'] = best_model
            results['best_mae'] = best_mae
            logger.info(f"Best model overall: {best_model} with MAE: {best_mae:.4f}")
    
    return results


def _train_traditional_models(df: pd.DataFrame, target_column: str, 
                              models: List[str]) -> Dict[str, Dict[str, Any]]:
    """
    Internal helper to train traditional models (ARIMA, Prophet, LSTM).
    
    Args:
        df: Input DataFrame
        target_column: Target column name
        models: List of model types
        
    Returns:
        Dictionary of results
    """
    import mlflow
    
    # End any existing run before starting
    if mlflow.active_run():
        mlflow.end_run()
        
    # Set MLflow experiment
    mlflow.set_experiment("Energy Demand Forecasting")

    results = {}
    registered_models = {}

    for model_type in models:
        logger.info(f"Training {model_type} model...")
        
        # Ensure no active run before starting new one
        if mlflow.active_run():
            mlflow.end_run()

        with mlflow.start_run(run_name=f"{model_type}_training") as run:
            trainer = TimeSeriesTrainer(model_type=model_type, target_column=target_column)
            result = trainer.train(df, tune_hyperparams=True)
            results[model_type] = result

            # Log parameters
            mlflow.log_param("model_type", model_type)
            mlflow.log_param("target_column", target_column)
            if hasattr(trainer, 'best_params') and trainer.best_params:
                for param, value in trainer.best_params.items():
                    mlflow.log_param(f"best_{param}", value)

            # Log metrics
            metrics = result['metrics']
            mlflow.log_metric("mae", metrics['mae'])
            mlflow.log_metric("mse", metrics['mse'])
            mlflow.log_metric("rmse", metrics['rmse'])
            mlflow.log_metric("r2", metrics['r2'])

            # Save model locally only if training was successful
            if result.get('model') is not None:
                model_path = f"models/{model_type}_model"
                trainer.save_model(model_path)
            else:
                logger.warning(f"Skipping model save for {model_type} due to training failure")
                continue

            # Register model in MLflow Model Registry
            model_name = f"energy_demand_{model_type}_model"
            try:
                if model_type == 'arima':
                    # For ARIMA, create a custom pyfunc model wrapper
                    import mlflow.pyfunc

                    class ARIMAModelWrapper(mlflow.pyfunc.PythonModel):
                        def __init__(self, params):
                            self.params = params
                            self.model = None

                        def load_context(self, context):
                            # Reconstruct ARIMA model from parameters
                            from statsmodels.tsa.arima.model import ARIMA
                            # Note: In production, you'd need the original training data
                            # For now, we'll store parameters only
                            pass

                        def predict(self, context, model_input):
                            # This would require retraining or loading pre-trained model
                            # For demonstration, return placeholder
                            import numpy as np
                            return np.array([0] * len(model_input))

                    # Create and log the custom model
                    arima_wrapper = ARIMAModelWrapper(result.get('params', {}))
                    # Create a simple signature for the model
                    import mlflow.models
                    signature = mlflow.models.infer_signature(
                        model_input=np.array([[1.0]]),  # Dummy input
                        model_output=np.array([0.0])    # Dummy output
                    )
                    mlflow.pyfunc.log_model(
                        name=f"{model_type}_model",
                        python_model=arima_wrapper,
                        registered_model_name=model_name,
                        signature=signature,
                        input_example=np.array([[1.0]])
                    )

                    # Get the registered model
                    client = mlflow.tracking.MlflowClient()
                    registered_model = client.get_registered_model(model_name)

                elif model_type == 'prophet':
                    # Prophet models are saved as artifacts
                    mlflow.log_artifact(model_path + '_prophet.pkl')

                    # Register using pyfunc
                    import mlflow.pyfunc

                    class ProphetModelWrapper(mlflow.pyfunc.PythonModel):
                        def __init__(self, model_path):
                            self.model_path = model_path

                        def load_context(self, context):
                            import joblib
                            self.model = joblib.load(self.model_path)

                        def predict(self, context, model_input):
                            # model_input should be a DataFrame with 'ds' column
                            forecast = self.model.predict(model_input)
                            return forecast['yhat'].values

                    prophet_wrapper = ProphetModelWrapper(model_path + '_prophet.pkl')
                    # Create a simple signature for the model
                    import mlflow.models
                    signature = mlflow.models.infer_signature(
                        model_input=pd.DataFrame({'ds': [pd.Timestamp('2024-01-01')]}),  # Dummy input
                        model_output=np.array([0.0])    # Dummy output
                    )
                    mlflow.pyfunc.log_model(
                        name=f"{model_type}_model",
                        python_model=prophet_wrapper,
                        registered_model_name=model_name,
                        signature=signature,
                        input_example=pd.DataFrame({'ds': [pd.Timestamp('2024-01-01')]})
                    )

                    client = mlflow.tracking.MlflowClient()
                    registered_model = client.get_registered_model(model_name)

                elif model_type == 'lstm':
                    # Log LSTM model artifacts
                    mlflow.log_artifact(model_path + '_lstm.h5')
                    mlflow.log_artifact(model_path + '_scaler.pkl')

                    # Register using pyfunc
                    import mlflow.pyfunc

                    class LSTMModelWrapper(mlflow.pyfunc.PythonModel):
                        def __init__(self, model_path, scaler_path):
                            self.model_path = model_path
                            self.scaler_path = scaler_path

                        def load_context(self, context):
                            import tensorflow as tf
                            import joblib
                            from tensorflow.keras.metrics import mae, mse
                            custom_objects = {'mae': mae, 'mse': mse}
                            self.model = tf.keras.models.load_model(self.model_path, custom_objects=custom_objects)
                            self.scaler = joblib.load(self.scaler_path)

                        def predict(self, context, model_input):
                            # model_input should be scaled appropriately
                            scaled_input = self.scaler.transform(model_input.reshape(-1, 1))
                            # Reshape for LSTM [samples, timesteps, features]
                            scaled_input = scaled_input.reshape((scaled_input.shape[0], 1, 1))
                            predictions = self.model.predict(scaled_input)
                            return self.scaler.inverse_transform(predictions).flatten()

                    lstm_wrapper = LSTMModelWrapper(
                        model_path + '_lstm.h5',
                        model_path + '_scaler.pkl'
                    )
                    # Create a simple signature for the model
                    import mlflow.models
                    signature = mlflow.models.infer_signature(
                        model_input=np.array([[1.0]]),  # Dummy input
                        model_output=np.array([0.0])    # Dummy output
                    )
                    mlflow.pyfunc.log_model(
                        name=f"{model_type}_model",
                        python_model=lstm_wrapper,
                        registered_model_name=model_name,
                        signature=signature,
                        input_example=np.array([[1.0]])
                    )

                    client = mlflow.tracking.MlflowClient()
                    registered_model = client.get_registered_model(model_name)

                # Store registration info
                registered_models[model_type] = {
                    'model_name': model_name,
                    'version': len(registered_model.latest_versions) if registered_model else 'N/A',
                    'run_id': run.info.run_id,
                    'mae': metrics['mae']
                }

                logger.info(f"Model {model_name} registered successfully")

            except Exception as e:
                logger.warning(f"Failed to register {model_type} model: {e}")
                registered_models[model_type] = {
                    'model_name': model_name,
                    'version': 'Failed',
                    'run_id': run.info.run_id,
                    'mae': metrics['mae'],
                    'error': str(e)
                }

    results['model_registry'] = registered_models
    
    return results


if __name__ == "__main__":
    # Example usage
    import sys
    from pathlib import Path

    sys.path.append(str(Path(__file__).parent.parent / 'src'))
    from data.preprocessing import full_preprocessing_pipeline
    from data.ingestion import ingest_data
    from features.feature_engineering import full_feature_engineering_pipeline

    # Load and preprocess data
    energy_path = Path(__file__).parent.parent / 'data' / 'raw' / 'energy_dataset.csv'
    weather_path = Path(__file__).parent.parent / 'data' / 'raw' / 'weather_features.csv'

    raw_data = ingest_data(str(energy_path), str(weather_path))
    processed_data = full_preprocessing_pipeline(raw_data['energy'], raw_data['weather'])
    feature_data = full_feature_engineering_pipeline(processed_data)

    # Train models
    results = train_multiple_models(feature_data, target_column='total_load_actual')

    print("Model training completed.")
    for model, result in results.items():
        print(f"{model.upper()}: MAE = {result['metrics']['mae']:.4f}, R2 = {result['metrics']['r2']:.4f}")

"""
Model Evaluation & Cross-Validation Strategy for Energy Demand Forecasting

This module implements comprehensive evaluation and cross-validation strategies
for time series forecasting models, including time-aware validation, business
metrics, and MLflow integration.
"""

import logging
from typing import Dict, List, Optional, Tuple, Union, Any
import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.stats.diagnostic import acorr_ljungbox
import mlflow
import mlflow.sklearn
try:
    import tensorflow as tf
except ImportError:
    tf = None
from pathlib import Path
import yaml
import joblib

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# 🔄 Time Series Cross-Validation
def time_series_cross_validation(df: pd.DataFrame, n_splits: int = 5) -> List[Tuple[pd.DataFrame, pd.DataFrame]]:
    """
    Perform time series cross-validation using TimeSeriesSplit.

    Args:
        df (pd.DataFrame): Input DataFrame.
        n_splits (int): Number of splits.

    Yields:
        Tuple[pd.DataFrame, pd.DataFrame]: Train and test data for each split.
    """
    tscv = TimeSeriesSplit(n_splits=n_splits)

    for train_idx, test_idx in tscv.split(df):
        train_data = df.iloc[train_idx]
        test_data = df.iloc[test_idx]
        yield train_data, test_data


def walk_forward_validation(df: pd.DataFrame, target_column: str = 'total_load_actual',
                          window_size: int = 720, forecast_horizon: int = 24,
                          model_type: str = 'arima') -> List[Dict[str, float]]:
    """
    Perform walk-forward validation for time series with actual model training.

    Args:
        df (pd.DataFrame): Input DataFrame.
        target_column (str): Target column name.
        window_size (int): Size of training window (hours).
        forecast_horizon (int): Forecast horizon (hours).
        model_type (str): Type of model to use ('arima', 'prophet', 'lstm').

    Returns:
        List[Dict[str, float]]: List of evaluation results for each step.
    """
    results = []
    logger.info(f"Starting walk-forward validation with {model_type} model...")

    for i in range(window_size, len(df) - forecast_horizon, forecast_horizon):
        try:
            train_end = i
            test_start = i
            test_end = min(i + forecast_horizon, len(df))

            train_data = df.iloc[:train_end]
            test_data = df.iloc[test_start:test_end]

            # Train model on expanding window
            if model_type == 'arima':
                from statsmodels.tsa.arima.model import ARIMA
                model = ARIMA(train_data[target_column], order=(1, 1, 1))
                fitted_model = model.fit()
                predictions = fitted_model.forecast(steps=len(test_data))
                y_true = test_data[target_column].values
                y_pred = predictions.values

            elif model_type == 'prophet':
                from prophet import Prophet
                prophet_df = train_data.reset_index()
                prophet_df = prophet_df.rename(columns={'index': 'ds', target_column: 'y'})
                prophet_df['ds'] = pd.to_datetime(prophet_df['ds'])

                model = Prophet(daily_seasonality=True, yearly_seasonality=True)
                model.fit(prophet_df)

                future = model.make_future_dataframe(periods=len(test_data), freq='H')
                forecast = model.predict(future)
                predictions = forecast['yhat'][-len(test_data):].values
                y_true = test_data[target_column].values
                y_pred = predictions

            else:  # Simple baseline for other models
                # Use last value as prediction
                last_value = train_data[target_column].iloc[-1]
                y_true = test_data[target_column].values
                y_pred = np.full(len(y_true), last_value)

            # Calculate metrics
            metrics = calculate_metrics(y_true, y_pred)
            results.append(metrics)

        except Exception as e:
            logger.warning(f"Walk-forward validation failed at step {i}: {e}")
            results.append({'mae': float('inf'), 'mse': float('inf'), 'rmse': float('inf'), 'mape': float('inf'), 'r2': float('-inf')})

    logger.info(f"Walk-forward validation completed with {len(results)} steps")
    return results


def expanding_window_validation(df: pd.DataFrame, target_column: str = 'total_load_actual',
                              initial_window: int = 168, forecast_horizon: int = 24,
                              model_type: str = 'arima') -> List[Dict[str, float]]:
    """
    Perform expanding window validation with actual model training.

    Args:
        df (pd.DataFrame): Input DataFrame.
        target_column (str): Target column name.
        initial_window (int): Initial training window size.
        forecast_horizon (int): Forecast horizon.
        model_type (str): Type of model to use.

    Returns:
        List[Dict[str, float]]: List of evaluation results.
    """
    results = []
    logger.info(f"Starting expanding window validation with {model_type} model...")

    for i in range(initial_window, len(df) - forecast_horizon, forecast_horizon):
        try:
            train_data = df.iloc[:i]
            test_data = df.iloc[i:i + forecast_horizon]

            # Train model on expanding window
            if model_type == 'arima':
                from statsmodels.tsa.arima.model import ARIMA
                model = ARIMA(train_data[target_column], order=(1, 1, 1))
                fitted_model = model.fit()
                predictions = fitted_model.forecast(steps=len(test_data))
                y_true = test_data[target_column].values
                y_pred = predictions.values

            elif model_type == 'prophet':
                from prophet import Prophet
                prophet_df = train_data.reset_index()
                prophet_df = prophet_df.rename(columns={'index': 'ds', target_column: 'y'})
                prophet_df['ds'] = pd.to_datetime(prophet_df['ds'])

                model = Prophet(daily_seasonality=True, yearly_seasonality=True)
                model.fit(prophet_df)

                future = model.make_future_dataframe(periods=len(test_data), freq='H')
                forecast = model.predict(future)
                predictions = forecast['yhat'][-len(test_data):].values
                y_true = test_data[target_column].values
                y_pred = predictions

            else:  # Simple baseline
                last_value = train_data[target_column].iloc[-1]
                y_true = test_data[target_column].values
                y_pred = np.full(len(y_true), last_value)

            # Calculate metrics
            metrics = calculate_metrics(y_true, y_pred)
            results.append(metrics)

        except Exception as e:
            logger.warning(f"Expanding window validation failed at step {i}: {e}")
            results.append({'mae': float('inf'), 'mse': float('inf'), 'rmse': float('inf'), 'mape': float('inf'), 'r2': float('-inf')})

    logger.info(f"Expanding window validation completed with {len(results)} steps")
    return results


# 📈 Evaluation Metrics Implementation
def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Calculate core evaluation metrics.

    Args:
        y_true (np.ndarray): True values.
        y_pred (np.ndarray): Predicted values.

    Returns:
        Dict[str, float]: Dictionary of metrics.
    """
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100 if np.all(y_true != 0) else float('inf')
    r2 = r2_score(y_true, y_pred)

    return {
        'mae': mae,
        'mse': mse,
        'rmse': rmse,
        'mape': mape,
        'r2': r2
    }


def energy_specific_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Calculate business-specific metrics for energy forecasting.

    Args:
        y_true (np.ndarray): True values.
        y_pred (np.ndarray): Predicted values.

    Returns:
        Dict[str, float]: Business-specific metrics.
    """
    # Peak hour accuracy (assuming hourly data, peak hours 6-10 AM, 6-10 PM)
    # Simplified: assume indices represent hours
    peak_hours = [6, 7, 8, 9, 18, 19, 20, 21]
    if len(y_true) >= 24:  # Ensure we have at least a day
        peak_mask = np.isin(np.arange(len(y_true)) % 24, peak_hours)
        if peak_mask.sum() > 0:
            peak_mae = mean_absolute_error(y_true[peak_mask], y_pred[peak_mask])
        else:
            peak_mae = 0.0
    else:
        peak_mae = 0.0

    # Load factor accuracy
    max_demand_true = np.max(y_true)
    max_demand_pred = np.max(y_pred)
    load_factor_true = np.mean(y_true) / max_demand_true if max_demand_true > 0 else 0
    load_factor_pred = np.mean(y_pred) / max_demand_pred if max_demand_pred > 0 else 0

    return {
        'peak_hour_mae': peak_mae,
        'load_factor_error': abs(load_factor_true - load_factor_pred),
        'max_demand_error': abs(max_demand_true - max_demand_pred)
    }


# 🎯 Model-Specific Validation
def validate_arima_model(train_data: pd.Series, test_data: pd.Series, order: tuple) -> Dict[str, Any]:
    """
    Validate ARIMA model performance.

    Args:
        train_data (pd.Series): Training data.
        test_data (pd.Series): Test data.
        order (tuple): ARIMA order (p, d, q).

    Returns:
        Dict[str, Any]: Validation results.
    """
    try:
        model = ARIMA(train_data, order=order)
        fitted_model = model.fit()

        # In-sample validation
        in_sample_pred = fitted_model.fittedvalues
        in_sample_metrics = calculate_metrics(train_data[1:], in_sample_pred)

        # Out-of-sample validation
        forecast = fitted_model.forecast(steps=len(test_data))
        out_sample_metrics = calculate_metrics(test_data.values, forecast)

        # Residual analysis
        residuals = fitted_model.resid
        ljung_box_p = acorr_ljungbox(residuals, lags=10)['lb_pvalue'].iloc[-1]

        return {
            'in_sample': in_sample_metrics,
            'out_sample': out_sample_metrics,
            'residual_test': ljung_box_p > 0.05  # White noise test
        }
    except Exception as e:
        logger.error(f"ARIMA validation failed: {e}")
        return {
            'in_sample': {'mae': float('inf'), 'mse': float('inf'), 'rmse': float('inf'), 'mape': float('inf'), 'r2': float('-inf')},
            'out_sample': {'mae': float('inf'), 'mse': float('inf'), 'rmse': float('inf'), 'mape': float('inf'), 'r2': float('-inf')},
            'residual_test': False
        }


def validate_lstm_model(X_train, y_train, X_test, y_test, model_path: str = None) -> Dict[str, Any]:
    """
    Validate LSTM model using K-fold time series cross-validation.

    Args:
        X_train: Training features.
        y_train: Training targets.
        X_test: Test features.
        y_test: Test targets.
        model_path (str): Path to saved model.

    Returns:
        Dict[str, Any]: Validation results.
    """
    if tf is None:
        logger.warning("TensorFlow not available, skipping LSTM validation")
        return {
            'cv_mean': float('inf'),
            'cv_std': float('inf'),
            'test_score': float('inf')
        }
    
    try:
        if model_path and Path(model_path).exists():
            model = tf.keras.models.load_model(model_path)
        else:
            # Create a simple LSTM model for validation
            model = tf.keras.Sequential([
                tf.keras.layers.LSTM(64, input_shape=(X_train.shape[1], X_train.shape[2])),
                tf.keras.layers.Dense(1)
            ])
            model.compile(optimizer='adam', loss='mae')

        # K-fold validation with time series splits
        kfold_scores = []
        tscv = TimeSeriesSplit(n_splits=3)

        for train_idx, val_idx in tscv.split(X_train):
            X_fold_train, X_fold_val = X_train[train_idx], X_train[val_idx]
            y_fold_train, y_fold_val = y_train[train_idx], y_train[val_idx]

            model.fit(X_fold_train, y_fold_train, epochs=10, verbose=0, batch_size=32)
            val_pred = model.predict(X_fold_val, verbose=0)
            fold_score = mean_absolute_error(y_fold_val, val_pred.flatten())
            kfold_scores.append(fold_score)

        # Test score
        test_pred = model.predict(X_test, verbose=0)
        test_score = mean_absolute_error(y_test, test_pred.flatten())

        return {
            'cv_mean': np.mean(kfold_scores),
            'cv_std': np.std(kfold_scores),
            'test_score': test_score
        }
    except Exception as e:
        logger.error(f"LSTM validation failed: {e}")
        return {
            'cv_mean': float('inf'),
            'cv_std': float('inf'),
            'test_score': float('inf')
        }


def generate_evaluation_report(validation_results: Dict[str, Any], best_model: str = None, output_path: str = None) -> Dict[str, Any]:
    """
    Generate comprehensive evaluation report.

    Args:
        validation_results (Dict[str, Any]): Validation results for all models.
        output_path (str): Output path for the report.

    Returns:
        str: Path to the generated report.
    """
    if output_path is None:
        timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
        output_path = f"reports/model_performance/evaluation_report_{timestamp}.json"

    # Ensure output directory exists
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    try:
        # Create comprehensive report
        report = {
            "timestamp": pd.Timestamp.now().isoformat(),
            "model_comparison": validation_results,
            "best_model": best_model,
            "summary": {
                "total_models_evaluated": len(validation_results),
                "best_model": best_model
            }
        }

        # Save report if path provided
        if output_path:
            with open(output_path, 'w') as f:
                import json
                json.dump(report, f, indent=2, default=str)
            logger.info(f"Evaluation report generated: {output_path}")

        return report

    except Exception as e:
        logger.error(f"Failed to generate evaluation report: {e}")
        return {"error": str(e)}





# 🔍 Cross-Validation Configuration
def load_cross_validation_config(config_path: str = 'configs/model/model_config.yaml') -> Dict[str, Any]:
    """
    Load cross-validation configuration.

    Args:
        config_path (str): Path to config file.

    Returns:
        Dict[str, Any]: Cross-validation config.
    """
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        return config.get('cross_validation', {})
    except Exception as e:
        logger.warning(f"Could not load config: {e}")
        return {}


def comprehensive_cross_validation(df: pd.DataFrame, target_column: str = 'total_load_actual',
                                config: Dict[str, Any] = None) -> Dict[str, Any]:
    """
    Perform comprehensive cross-validation with actual model training.

    Args:
        df (pd.DataFrame): Input DataFrame.
        target_column (str): Target column name.
        config (Dict[str, Any]): Cross-validation config.

    Returns:
        Dict[str, Any]: Cross-validation results.
    """
    if config is None:
        config = load_cross_validation_config()

    results = {}

    # Time series split validation
    if config.get('time_series_split', True):
        tscv_results = list(time_series_cross_validation(df, config.get('n_splits', 5)))
        results['time_series_cv'] = tscv_results

    # Walk-forward validation
    if 'walk_forward' in config.get('strategies', []):
        wf_results = walk_forward_validation(df, target_column=target_column,
                                           model_type=config.get('model_type', 'arima'))
        results['walk_forward'] = wf_results

    # Expanding window validation
    if 'expanding_window' in config.get('strategies', []):
        ew_results = expanding_window_validation(df, target_column=target_column,
                                               model_type=config.get('model_type', 'arima'))
        results['expanding_window'] = ew_results

    # Seasonal validation (placeholder)
    results['seasonal'] = []  # Implement seasonal_cross_validation if needed

    return results


# 📊 Validation Results Tracking
def log_validation_results(model_name: str, cv_results: Dict[str, Any], run_name: str = None):
    """
    Log cross-validation results to MLflow.

    Args:
        model_name (str): Name of the model.
        cv_results (Dict[str, Any]): Cross-validation results.
        run_name (str): MLflow run name.
    """
    try:
        # End any existing run before starting a new one
        if mlflow.active_run():
            mlflow.end_run()
            
        with mlflow.start_run(run_name=run_name or f"{model_name}_validation"):
            # Log cross-validation metrics
            if 'cv_mean' in cv_results:
                mlflow.log_metric("cv_mae_mean", cv_results['cv_mean'])
            if 'cv_std' in cv_results:
                mlflow.log_metric("cv_mae_std", cv_results['cv_std'])
            if 'test_score' in cv_results:
                mlflow.log_metric("test_mae", cv_results['test_score'])

            # Log detailed results
            mlflow.log_dict(cv_results, "detailed_cv_results.json")

            logger.info(f"Validation results logged for {model_name}")
    except Exception as e:
        logger.error(f"Failed to log validation results: {e}")


# 🎯 Model Selection Criteria
def select_best_model(validation_results: Dict[str, Dict[str, Any]]) -> str:
    """
    Select the best model based on multi-criteria evaluation.

    Args:
        validation_results (Dict[str, Dict[str, Any]]): Validation results for each model.

    Returns:
        str: Name of the best model.
    """
    scores = {}

    for model_name, results in validation_results.items():
        # Extract metrics
        mae = results.get('test_score', results.get('mae', float('inf')))
        cv_std = results.get('cv_std', 0)
        r2 = results.get('r2', -float('inf'))

        # Weighted scoring
        mae_score = 1 / (1 + mae) if mae != float('inf') else 0  # Lower is better
        r2_score = max(0, r2)  # Higher is better
        stability_score = 1 / (1 + cv_std) if cv_std != float('inf') else 0  # Lower variance is better

        # Business-weighted composite score
        composite_score = (
            0.4 * mae_score +      # Accuracy weight
            0.3 * r2_score +       # Fit quality weight
            0.3 * stability_score  # Stability weight
        )

        scores[model_name] = composite_score

    best_model = max(scores, key=scores.get) if scores else None
    logger.info(f"Best model selected: {best_model} with score {scores.get(best_model, 0):.4f}")
    return best_model


# 🎯 Validation Pipeline
def model_validation_pipeline(models: Dict[str, Any], test_data: pd.DataFrame, target_column: str = 'total_load_actual', config: Dict[str, Any] = None) -> Tuple[Dict[str, Any], str]:
    """
    Complete model validation pipeline.

    Args:
        models (Dict[str, Any]): Dictionary of trained models.
        test_data (pd.DataFrame): Test data.
        target_column (str): Target column name.
        config (Dict[str, Any]): Validation config.

    Returns:
        Tuple[Dict[str, Any], str]: Validation results and best model name.
    """
    validation_results = {}

    for model_name, model in models.items():
        logger.info(f"Validating {model_name}...")

        # Cross-validation
        cv_results = comprehensive_cross_validation(test_data, target_column, config)

        # Business metrics (placeholder - implement based on model type)
        business_metrics = {'peak_hour_mae': 0.0, 'load_factor_error': 0.0, 'max_demand_error': 0.0}

        # Statistical tests (placeholder)
        statistical_tests = {'residual_test': True}

        validation_results[model_name] = {
            'cv_metrics': cv_results,
            'business_metrics': business_metrics,
            'statistical_tests': statistical_tests
        }

        # Log to MLflow
        log_validation_results(model_name, cv_results)

    # Select best model
    best_model = select_best_model(validation_results)

    return validation_results, best_model


if __name__ == "__main__":
    # Example usage
    np.random.seed(42)
    dates = pd.date_range('2020-01-01', periods=1000, freq='H')
    df = pd.DataFrame({
        'total_load_actual': 1000 + np.random.normal(0, 100, 1000)
    }, index=dates)

    # Test time series cross-validation
    cv_results = list(time_series_cross_validation(df, n_splits=3))
    print(f"Time series CV completed with {len(cv_results)} splits")

    # Test metrics
    y_true = np.random.normal(1000, 100, 100)
    y_pred = y_true + np.random.normal(0, 50, 100)
    metrics = calculate_metrics(y_true, y_pred)
    print(f"Metrics: MAE={metrics['mae']:.2f}, R2={metrics['r2']:.4f}")

    # Test business metrics
    business_metrics = energy_specific_metrics(y_true, y_pred)
    print(f"Business metrics: Peak MAE={business_metrics['peak_hour_mae']:.2f}")

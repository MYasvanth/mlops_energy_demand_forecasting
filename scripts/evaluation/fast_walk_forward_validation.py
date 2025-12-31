#!/usr/bin/env python3
"""
Fast Walk-Forward Validation for Energy Demand Forecasting
Optimized for speed with parallel processing and reduced data sampling
"""

import pandas as pd
import numpy as np
import logging
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Dict, List, Any, Tuple
import time
import mlflow
from pathlib import Path
import sys
import os
import warnings

# Suppress warnings
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TensorFlow warnings
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # Disable oneDNN warnings

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Fast mode configuration
FAST_CONFIG = {
    'n_splits': 3,
    'test_size': 12,
    'sample_ratio': 1.0,  # Use full dataset for better model training
    'max_trials': 20,
    'parallel': True,
    'max_workers': 4
}

def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Calculate evaluation metrics quickly"""
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100 if np.all(y_true != 0) else float('inf')
    r2 = r2_score(y_true, y_pred)
    
    return {'mae': mae, 'mse': mse, 'rmse': rmse, 'mape': mape, 'r2': r2}

def fast_walk_forward_validation(data: pd.DataFrame, model_type: str, 
                                target_column: str = 'total_load_actual',
                                config: Dict = None) -> List[Dict[str, float]]:
    """Fast walk-forward validation with reduced splits and data sampling"""
    if config is None:
        config = FAST_CONFIG
    
    # Sample data for speed
    if config['sample_ratio'] < 1.0:
        sample_size = int(len(data) * config['sample_ratio'])
        data = data.tail(sample_size)  # Use recent data
    
    results = []
    n_splits = config['n_splits']
    test_size = config['test_size']
    
    logger.info(f"Running fast walk-forward validation for {model_type} with {n_splits} splits")
    
    for i in range(n_splits):
        try:
            # Calculate split indices
            total_size = len(data)
            train_end = total_size - (n_splits - i) * test_size
            test_start = train_end
            test_end = test_start + test_size
            
            if train_end < 50:  # Minimum training size
                continue
                
            train_data = data.iloc[:train_end]
            test_data = data.iloc[test_start:test_end]
            
            # Quick model training and prediction
            if model_type == 'arima':
                metrics = _train_arima_fast(train_data, test_data, target_column)
            elif model_type == 'prophet':
                metrics = _train_prophet_fast(train_data, test_data, target_column)
            elif model_type == 'lstm':
                metrics = _train_lstm_fast(train_data, test_data, target_column)
            else:
                # Baseline model
                metrics = _train_baseline(train_data, test_data, target_column)
            
            results.append(metrics)
            
        except Exception as e:
            logger.warning(f"Split {i} failed: {e}")
            results.append({'mae': float('inf'), 'mse': float('inf'), 'rmse': float('inf'), 'mape': float('inf'), 'r2': -1})
    
    return results

def _train_arima_fast(train_data: pd.DataFrame, test_data: pd.DataFrame, target_column: str) -> Dict[str, float]:
    """Fast ARIMA training with simple order"""
    try:
        from statsmodels.tsa.arima.model import ARIMA
        
        # Create proper time series with frequency
        ts_data = train_data[target_column].copy()
        if not isinstance(ts_data.index, pd.DatetimeIndex):
            ts_data.index = pd.to_datetime(ts_data.index)
        
        # Infer frequency if not set
        if ts_data.index.freq is None:
            ts_data = ts_data.asfreq('H')  # Assume hourly frequency
        
        # Use simple order for speed
        model = ARIMA(ts_data, order=(1, 1, 1))
        fitted_model = model.fit()
        
        predictions = fitted_model.forecast(steps=len(test_data))
        y_true = test_data[target_column].values
        y_pred = predictions.values if hasattr(predictions, 'values') else predictions
        
        return calculate_metrics(y_true, y_pred)
    except Exception as e:
        logger.warning(f"ARIMA training failed: {e}")
        return {'mae': float('inf'), 'mse': float('inf'), 'rmse': float('inf'), 'mape': float('inf'), 'r2': -1}

def _train_prophet_fast(train_data: pd.DataFrame, test_data: pd.DataFrame, target_column: str) -> Dict[str, float]:
    """Fast Prophet training with minimal settings"""
    try:
        from prophet import Prophet
        
        # Prepare data with proper column names
        prophet_df = train_data.reset_index()
        
        # Ensure we have the right column names
        if 'index' in prophet_df.columns:
            prophet_df = prophet_df.rename(columns={'index': 'ds', target_column: 'y'})
        else:
            # If index is already a column name, use the first column as ds
            cols = prophet_df.columns.tolist()
            prophet_df = prophet_df.rename(columns={cols[0]: 'ds', target_column: 'y'})
        
        # Ensure ds column exists and is datetime
        if 'ds' not in prophet_df.columns:
            raise ValueError("Could not create 'ds' column for Prophet")
            
        prophet_df['ds'] = pd.to_datetime(prophet_df['ds'])
        
        # Keep only ds and y columns
        prophet_df = prophet_df[['ds', 'y']].dropna()
        
        # Fast Prophet settings
        model = Prophet(
            daily_seasonality=True,
            yearly_seasonality=False,
            weekly_seasonality=True,
            uncertainty_samples=0,  # Disable uncertainty for speed
            mcmc_samples=0  # Disable MCMC for speed
        )
        model.fit(prophet_df)
        
        future = model.make_future_dataframe(periods=len(test_data), freq='H')
        forecast = model.predict(future)
        predictions = forecast['yhat'][-len(test_data):].values
        
        y_true = test_data[target_column].values
        return calculate_metrics(y_true, predictions)
        
    except Exception as e:
        logger.warning(f"Prophet training failed: {e}")
        return {'mae': float('inf'), 'mse': float('inf'), 'rmse': float('inf'), 'mape': float('inf'), 'r2': -1}

def _train_lstm_fast(train_data: pd.DataFrame, test_data: pd.DataFrame, target_column: str) -> Dict[str, float]:
    """Fast LSTM training with minimal epochs"""
    try:
        import tensorflow as tf
        
        # Prepare sequences
        def create_sequences(data, seq_length=24):
            X, y = [], []
            for i in range(seq_length, len(data)):
                X.append(data[i-seq_length:i])
                y.append(data[i])
            return np.array(X), np.array(y)
        
        train_values = train_data[target_column].values
        test_values = test_data[target_column].values
        
        # Normalize
        mean_val = np.mean(train_values)
        std_val = np.std(train_values)
        train_norm = (train_values - mean_val) / std_val
        
        X_train, y_train = create_sequences(train_norm)
        
        if len(X_train) < 10:  # Not enough data
            return {'mae': float('inf'), 'mse': float('inf'), 'rmse': float('inf'), 'mape': float('inf'), 'r2': -1}
        
        # Simple LSTM model
        model = tf.keras.Sequential([
            tf.keras.layers.LSTM(32, input_shape=(X_train.shape[1], 1)),
            tf.keras.layers.Dense(1)
        ])
        model.compile(optimizer='adam', loss='mae')
        
        # Fast training with more epochs for better learning
        X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
        model.fit(X_train, y_train, epochs=20, verbose=0, batch_size=32)
        
        # Predict
        last_sequence = train_norm[-24:].reshape(1, 24, 1)
        predictions = []
        
        for _ in range(len(test_data)):
            pred = model.predict(last_sequence, verbose=0)[0, 0]
            predictions.append(pred * std_val + mean_val)
            # Update sequence (simplified)
            last_sequence = np.roll(last_sequence, -1, axis=1)
            last_sequence[0, -1, 0] = pred
        
        y_true = test_values
        y_pred = np.array(predictions)
        
        return calculate_metrics(y_true, y_pred)
        
    except Exception as e:
        logger.warning(f"LSTM training failed: {e}")
        return {'mae': float('inf'), 'mse': float('inf'), 'rmse': float('inf'), 'mape': float('inf'), 'r2': -1}

def _train_baseline(train_data: pd.DataFrame, test_data: pd.DataFrame, target_column: str) -> Dict[str, float]:
    """Baseline model using last value"""
    try:
        last_value = train_data[target_column].iloc[-1]
        y_true = test_data[target_column].values
        y_pred = np.full(len(y_true), last_value)
        
        return calculate_metrics(y_true, y_pred)
    except Exception as e:
        logger.warning(f"Baseline training failed: {e}")
        return {'mae': float('inf'), 'mse': float('inf'), 'rmse': float('inf'), 'mape': float('inf'), 'r2': -1}

def run_single_model(args):
    """Run validation for a single model (for parallel processing)"""
    model_type, data, target_column, config = args
    return model_type, fast_walk_forward_validation(data, model_type, target_column, config)

def run_all_models_parallel(data: pd.DataFrame, target_column: str = 'total_load_actual', 
                           models: List[str] = None, config: Dict = None) -> Dict[str, Any]:
    """Run all models in parallel for maximum speed"""
    if models is None:
        models = ['baseline', 'arima', 'prophet', 'lstm']
    
    if config is None:
        config = FAST_CONFIG
    
    logger.info(f"Running {len(models)} models in parallel...")
    start_time = time.time()
    
    results = {}
    
    if config.get('parallel', True):
        # Parallel execution
        with ProcessPoolExecutor(max_workers=config.get('max_workers', 4)) as executor:
            # Submit all jobs
            future_to_model = {
                executor.submit(run_single_model, (model, data, target_column, config)): model 
                for model in models
            }
            
            # Collect results
            for future in as_completed(future_to_model):
                try:
                    model_type, model_results = future.result()
                    results[model_type] = model_results
                    logger.info(f"Completed {model_type}")
                except Exception as e:
                    model_type = future_to_model[future]
                    logger.error(f"Model {model_type} failed: {e}")
                    results[model_type] = [{'mae': float('inf'), 'mse': float('inf'), 'rmse': float('inf'), 'mape': float('inf'), 'r2': -1}]
    else:
        # Sequential execution
        for model in models:
            logger.info(f"Running {model}...")
            results[model] = fast_walk_forward_validation(data, model, target_column, config)
    
    elapsed_time = time.time() - start_time
    logger.info(f"All models completed in {elapsed_time:.2f} seconds")
    
    return results

def summarize_results(results: Dict[str, List[Dict[str, float]]]) -> Dict[str, Dict[str, float]]:
    """Summarize validation results"""
    summary = {}
    
    for model_name, model_results in results.items():
        if not model_results:
            continue
            
        # Calculate averages
        metrics = ['mae', 'mse', 'rmse', 'mape', 'r2']
        avg_metrics = {}
        
        for metric in metrics:
            values = [r[metric] for r in model_results if r[metric] != float('inf') and r[metric] != float('-inf')]
            if values:
                avg_metrics[f'{metric}_mean'] = np.mean(values)
                avg_metrics[f'{metric}_std'] = np.std(values)
            else:
                avg_metrics[f'{metric}_mean'] = float('inf')
                avg_metrics[f'{metric}_std'] = 0
        
        summary[model_name] = avg_metrics
    
    return summary

def select_best_model(summary: Dict[str, Dict[str, float]]) -> str:
    """Select best model based on MAE"""
    best_model = None
    best_mae = float('inf')
    
    for model_name, metrics in summary.items():
        mae = metrics.get('mae_mean', float('inf'))
        if mae < best_mae:
            best_mae = mae
            best_model = model_name
    
    return best_model

def main():
    """Main execution function"""
    # Load data
    data_path = "data/processed/processed_energy_weather.csv"
    
    if not os.path.exists(data_path):
        logger.error(f"Data file not found: {data_path}")
        # Create sample data for demo
        logger.info("Creating sample data for demonstration...")
        dates = pd.date_range('2020-01-01', periods=1000, freq='H')
        data = pd.DataFrame({
            'total_load_actual': 1000 + 200 * np.sin(np.arange(1000) * 2 * np.pi / 24) + np.random.normal(0, 50, 1000)
        }, index=dates)
    else:
        # Fix date parsing warning with fallback
        try:
            data = pd.read_csv(data_path, index_col=0, parse_dates=True, date_format='%Y-%m-%d %H:%M:%S')
        except:
            # Fallback to automatic parsing if format doesn't match
            data = pd.read_csv(data_path, index_col=0)
            data.index = pd.to_datetime(data.index, errors='coerce')
    
    logger.info(f"Data shape: {data.shape}")
    
    # Run fast validation
    models_to_test = ['baseline', 'arima']  # Start with fast models
    
    try:
        # Add Prophet if available
        import prophet
        models_to_test.append('prophet')
    except ImportError:
        logger.warning("Prophet not available")
    
    try:
        # Add LSTM if TensorFlow available
        import tensorflow
        models_to_test.append('lstm')
    except ImportError:
        logger.warning("TensorFlow not available")
    
    logger.info(f"Testing models: {models_to_test}")
    
    # Run validation
    results = run_all_models_parallel(data, models=models_to_test, config=FAST_CONFIG)
    
    # Summarize results
    summary = summarize_results(results)
    
    # Select best model
    best_model = select_best_model(summary)
    
    # Print results
    print("\n" + "="*50)
    print("FAST WALK-FORWARD VALIDATION RESULTS")
    print("="*50)
    
    for model_name, metrics in summary.items():
        print(f"\n{model_name.upper()}:")
        print(f"  MAE: {metrics['mae_mean']:.2f} ± {metrics['mae_std']:.2f}")
        print(f"  RMSE: {metrics['rmse_mean']:.2f} ± {metrics['rmse_std']:.2f}")
        print(f"  R²: {metrics['r2_mean']:.4f} ± {metrics['r2_std']:.4f}")
    
    print(f"\nBEST MODEL: {best_model}")
    print("="*50)
    
    # Log to MLflow
    try:
        if mlflow.active_run():
            mlflow.end_run()
            
        with mlflow.start_run(run_name="fast_walk_forward_validation"):
            for model_name, metrics in summary.items():
                mlflow.log_metric(f"{model_name}_mae_mean", metrics['mae_mean'])
                mlflow.log_metric(f"{model_name}_rmse_mean", metrics['rmse_mean'])
                mlflow.log_metric(f"{model_name}_r2_mean", metrics['r2_mean'])
            
            mlflow.log_param("best_model", best_model)
            mlflow.log_param("validation_type", "fast_walk_forward")
            mlflow.log_param("n_splits", FAST_CONFIG['n_splits'])
            
        logger.info("Results logged to MLflow")
    except Exception as e:
        logger.warning(f"MLflow logging failed: {e}")

if __name__ == "__main__":
    main()
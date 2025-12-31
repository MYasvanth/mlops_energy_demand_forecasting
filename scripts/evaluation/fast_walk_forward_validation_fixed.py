#!/usr/bin/env python3
"""
Fixed Fast Walk-Forward Validation for Energy Demand Forecasting
Addresses date parsing, ARIMA index, and Prophet column issues
"""

import pandas as pd
import numpy as np
import logging
import warnings
import os
from typing import Dict, List, Any

# Suppress warnings
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Calculate evaluation metrics"""
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100 if np.all(y_true != 0) else float('inf')
    r2 = r2_score(y_true, y_pred)
    
    return {'mae': mae, 'mse': mse, 'rmse': rmse, 'mape': mape, 'r2': r2}

def load_data_safely(data_path: str) -> pd.DataFrame:
    """Load data with proper date parsing"""
    if not os.path.exists(data_path):
        logger.warning(f"Data file not found: {data_path}. Creating sample data.")
        dates = pd.date_range('2020-01-01', periods=1000, freq='H')
        return pd.DataFrame({
            'total_load_actual': 1000 + 200 * np.sin(np.arange(1000) * 2 * np.pi / 24) + np.random.normal(0, 50, 1000)
        }, index=dates)
    
    try:
        # Try with inferred date format first
        data = pd.read_csv(data_path, index_col=0)
        data.index = pd.to_datetime(data.index, errors='coerce')
        return data
    except Exception as e:
        logger.error(f"Failed to load data: {e}")
        raise

def train_arima_fixed(train_data: pd.DataFrame, test_data: pd.DataFrame, target_column: str) -> Dict[str, float]:
    """Fixed ARIMA training with proper time series handling"""
    try:
        from statsmodels.tsa.arima.model import ARIMA
        
        # Prepare time series - handle duplicate index issue
        ts_data = train_data[target_column].copy()
        
        # Remove duplicates in index
        ts_data = ts_data[~ts_data.index.duplicated(keep='first')]
        
        # Ensure datetime index
        if not isinstance(ts_data.index, pd.DatetimeIndex):
            ts_data.index = pd.to_datetime(ts_data.index)
        
        # Sort by index to ensure proper time order
        ts_data = ts_data.sort_index()
        
        # Remove any NaN values
        ts_data = ts_data.dropna()
        
        if len(ts_data) < 50:  # Need more data for ARIMA
            raise ValueError("Insufficient data for ARIMA")
        
        # Use simple differencing instead of integrated ARIMA
        model = ARIMA(ts_data, order=(1, 0, 1))  # Changed from (1,1,1) to (1,0,1)
        fitted_model = model.fit()
        
        # Make predictions
        predictions = fitted_model.forecast(steps=len(test_data))
        y_true = test_data[target_column].values
        y_pred = predictions.values if hasattr(predictions, 'values') else predictions
        
        return calculate_metrics(y_true, y_pred)
        
    except Exception as e:
        logger.warning(f"ARIMA training failed: {e}")
        return {'mae': float('inf'), 'mse': float('inf'), 'rmse': float('inf'), 'mape': float('inf'), 'r2': -1}

def train_prophet_fixed(train_data: pd.DataFrame, test_data: pd.DataFrame, target_column: str) -> Dict[str, float]:
    """Fixed Prophet training with proper data preparation"""
    try:
        from prophet import Prophet
        
        # Create Prophet dataframe with proper handling
        train_clean = train_data[target_column].dropna()
        train_clean = train_clean[~train_clean.index.duplicated(keep='first')]
        
        prophet_df = pd.DataFrame()
        prophet_df['ds'] = pd.to_datetime(train_clean.index)
        prophet_df['y'] = train_clean.values
        
        # Remove any remaining NaN values and ensure proper data types
        prophet_df = prophet_df.dropna()
        prophet_df['y'] = prophet_df['y'].astype(float)
        
        if len(prophet_df) < 50:
            raise ValueError("Insufficient data for Prophet")
        
        # Scale down the data to prevent overflow
        y_mean = prophet_df['y'].mean()
        y_std = prophet_df['y'].std()
        prophet_df['y'] = (prophet_df['y'] - y_mean) / y_std
        
        # Initialize Prophet with simpler settings
        model = Prophet(
            daily_seasonality=False,  # Disable to prevent overfitting
            yearly_seasonality=False,
            weekly_seasonality=False,
            uncertainty_samples=0,
            mcmc_samples=0
        )
        
        model.fit(prophet_df)
        
        # Create future dataframe
        future = model.make_future_dataframe(periods=len(test_data), freq='H')
        forecast = model.predict(future)
        
        # Extract and rescale predictions
        predictions = forecast['yhat'][-len(test_data):].values
        predictions = predictions * y_std + y_mean  # Rescale back
        
        y_true = test_data[target_column].values
        
        return calculate_metrics(y_true, predictions)
        
    except Exception as e:
        logger.warning(f"Prophet training failed: {e}")
        return {'mae': float('inf'), 'mse': float('inf'), 'rmse': float('inf'), 'mape': float('inf'), 'r2': -1}

def train_lstm_fixed(train_data: pd.DataFrame, test_data: pd.DataFrame, target_column: str) -> Dict[str, float]:
    """Fixed LSTM training with proper sequence handling"""
    try:
        import tensorflow as tf
        tf.get_logger().setLevel('ERROR')
        
        # Prepare data
        train_values = train_data[target_column].dropna().values
        
        if len(train_values) < 100:
            raise ValueError("Insufficient data for LSTM")
        
        # Normalize data
        mean_val = np.mean(train_values)
        std_val = np.std(train_values)
        train_norm = (train_values - mean_val) / std_val
        
        # Create sequences
        seq_length = 24
        X, y = [], []
        for i in range(seq_length, len(train_norm)):
            X.append(train_norm[i-seq_length:i])
            y.append(train_norm[i])
        
        X, y = np.array(X), np.array(y)
        X = X.reshape((X.shape[0], X.shape[1], 1))
        
        # Simple LSTM model
        model = tf.keras.Sequential([
            tf.keras.layers.LSTM(32, input_shape=(seq_length, 1)),
            tf.keras.layers.Dense(1)
        ])
        model.compile(optimizer='adam', loss='mae')
        
        # Train model
        model.fit(X, y, epochs=10, verbose=0, batch_size=32)
        
        # Make predictions
        last_seq = train_norm[-seq_length:].reshape(1, seq_length, 1)
        predictions = []
        
        for _ in range(len(test_data)):
            pred = model.predict(last_seq, verbose=0)[0, 0]
            predictions.append(pred * std_val + mean_val)
            # Update sequence
            last_seq = np.roll(last_seq, -1, axis=1)
            last_seq[0, -1, 0] = pred
        
        y_true = test_data[target_column].values
        y_pred = np.array(predictions)
        
        return calculate_metrics(y_true, y_pred)
        
    except Exception as e:
        logger.warning(f"LSTM training failed: {e}")
        return {'mae': float('inf'), 'mse': float('inf'), 'rmse': float('inf'), 'mape': float('inf'), 'r2': -1}

def train_baseline(train_data: pd.DataFrame, test_data: pd.DataFrame, target_column: str) -> Dict[str, float]:
    """Baseline model using last value"""
    try:
        last_value = train_data[target_column].iloc[-1]
        y_true = test_data[target_column].values
        y_pred = np.full(len(y_true), last_value)
        
        return calculate_metrics(y_true, y_pred)
    except Exception as e:
        logger.warning(f"Baseline training failed: {e}")
        return {'mae': float('inf'), 'mse': float('inf'), 'rmse': float('inf'), 'mape': float('inf'), 'r2': -1}

def run_validation(data: pd.DataFrame, target_column: str = 'total_load_actual') -> Dict[str, List[Dict[str, float]]]:
    """Run walk-forward validation for all models"""
    
    results = {}
    n_splits = 3
    test_size = 12
    
    models = {
        'baseline': train_baseline,
        'arima': train_arima_fixed,
        'prophet': train_prophet_fixed,
        'lstm': train_lstm_fixed
    }
    
    for model_name, model_func in models.items():
        logger.info(f"Running {model_name} validation...")
        model_results = []
        
        for i in range(n_splits):
            try:
                # Calculate split indices
                total_size = len(data)
                train_end = total_size - (n_splits - i) * test_size
                test_start = train_end
                test_end = test_start + test_size
                
                if train_end < 50:
                    continue
                    
                train_data = data.iloc[:train_end]
                test_data = data.iloc[test_start:test_end]
                
                # Train and evaluate model
                metrics = model_func(train_data, test_data, target_column)
                model_results.append(metrics)
                
            except Exception as e:
                logger.warning(f"{model_name} split {i} failed: {e}")
                model_results.append({'mae': float('inf'), 'mse': float('inf'), 'rmse': float('inf'), 'mape': float('inf'), 'r2': -1})
        
        results[model_name] = model_results
        logger.info(f"Completed {model_name}")
    
    return results

def summarize_results(results: Dict[str, List[Dict[str, float]]]) -> Dict[str, Dict[str, float]]:
    """Summarize validation results"""
    summary = {}
    
    for model_name, model_results in results.items():
        if not model_results:
            continue
            
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

def main():
    """Main execution function"""
    # Load data
    data_path = "data/processed/processed_energy_weather.csv"
    data = load_data_safely(data_path)
    
    logger.info(f"Data shape: {data.shape}")
    
    # Run validation
    results = run_validation(data)
    
    # Summarize results
    summary = summarize_results(results)
    
    # Print results
    print("\n" + "="*50)
    print("FIXED WALK-FORWARD VALIDATION RESULTS")
    print("="*50)
    
    for model_name, metrics in summary.items():
        print(f"\n{model_name.upper()}:")
        print(f"  MAE: {metrics['mae_mean']:.2f} ± {metrics['mae_std']:.2f}")
        print(f"  RMSE: {metrics['rmse_mean']:.2f} ± {metrics['rmse_std']:.2f}")
        print(f"  R²: {metrics['r2_mean']:.4f} ± {metrics['r2_std']:.4f}")
    
    # Find best model
    best_model = min(summary.keys(), key=lambda x: summary[x]['mae_mean'])
    print(f"\nBEST MODEL: {best_model}")
    print("="*50)

if __name__ == "__main__":
    main()
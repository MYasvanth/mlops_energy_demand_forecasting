#!/usr/bin/env python3
"""
Run All Models with Fast Walk-Forward Validation
Forces execution of all available models
"""

import pandas as pd
import numpy as np
import logging
import time
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Calculate evaluation metrics"""
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100 if np.all(y_true != 0) else float('inf')
    r2 = r2_score(y_true, y_pred)
    
    return {'mae': mae, 'mse': mse, 'rmse': rmse, 'mape': mape, 'r2': r2}

def run_baseline_model(train_data: pd.DataFrame, test_data: pd.DataFrame, target_column: str) -> dict:
    """Baseline model using last value"""
    last_value = train_data[target_column].iloc[-1]
    y_true = test_data[target_column].values
    y_pred = np.full(len(y_true), last_value)
    return calculate_metrics(y_true, y_pred)

def run_arima_model(train_data: pd.DataFrame, test_data: pd.DataFrame, target_column: str) -> dict:
    """ARIMA model"""
    try:
        from statsmodels.tsa.arima.model import ARIMA
        model = ARIMA(train_data[target_column], order=(1, 1, 1))
        fitted_model = model.fit()
        predictions = fitted_model.forecast(steps=len(test_data))
        y_true = test_data[target_column].values
        y_pred = predictions.values if hasattr(predictions, 'values') else predictions
        return calculate_metrics(y_true, y_pred)
    except Exception as e:
        logger.warning(f"ARIMA failed: {e}")
        return {'mae': float('inf'), 'mse': float('inf'), 'rmse': float('inf'), 'mape': float('inf'), 'r2': -1}

def run_prophet_model(train_data: pd.DataFrame, test_data: pd.DataFrame, target_column: str) -> dict:
    """Prophet model"""
    try:
        from prophet import Prophet
        
        # Prepare data
        prophet_df = train_data.reset_index()
        prophet_df = prophet_df.rename(columns={'index': 'ds', target_column: 'y'})
        prophet_df['ds'] = pd.to_datetime(prophet_df['ds'])
        
        # Fast Prophet settings
        model = Prophet(
            daily_seasonality=True,
            yearly_seasonality=False,
            weekly_seasonality=True,
            uncertainty_samples=0,
            mcmc_samples=0
        )
        model.fit(prophet_df)
        
        future = model.make_future_dataframe(periods=len(test_data), freq='H')
        forecast = model.predict(future)
        predictions = forecast['yhat'][-len(test_data):].values
        
        y_true = test_data[target_column].values
        return calculate_metrics(y_true, predictions)
        
    except ImportError:
        logger.warning("Prophet not installed")
        return {'mae': float('inf'), 'mse': float('inf'), 'rmse': float('inf'), 'mape': float('inf'), 'r2': -1}
    except Exception as e:
        logger.warning(f"Prophet failed: {e}")
        return {'mae': float('inf'), 'mse': float('inf'), 'rmse': float('inf'), 'mape': float('inf'), 'r2': -1}

def run_lstm_model(train_data: pd.DataFrame, test_data: pd.DataFrame, target_column: str) -> dict:
    """LSTM model"""
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
        
        if len(X_train) < 10:
            return {'mae': float('inf'), 'mse': float('inf'), 'rmse': float('inf'), 'mape': float('inf'), 'r2': -1}
        
        # Simple LSTM model
        model = tf.keras.Sequential([
            tf.keras.layers.LSTM(32, input_shape=(X_train.shape[1], 1)),
            tf.keras.layers.Dense(1)
        ])
        model.compile(optimizer='adam', loss='mae')
        
        # Fast training
        X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
        model.fit(X_train, y_train, epochs=5, verbose=0, batch_size=32)
        
        # Predict
        last_sequence = train_norm[-24:].reshape(1, 24, 1)
        predictions = []
        
        for _ in range(len(test_data)):
            pred = model.predict(last_sequence, verbose=0)[0, 0]
            predictions.append(pred * std_val + mean_val)
            last_sequence = np.roll(last_sequence, -1, axis=1)
            last_sequence[0, -1, 0] = pred
        
        y_true = test_values
        y_pred = np.array(predictions)
        
        return calculate_metrics(y_true, y_pred)
        
    except ImportError:
        logger.warning("TensorFlow not installed")
        return {'mae': float('inf'), 'mse': float('inf'), 'rmse': float('inf'), 'mape': float('inf'), 'r2': -1}
    except Exception as e:
        logger.warning(f"LSTM failed: {e}")
        return {'mae': float('inf'), 'mse': float('inf'), 'rmse': float('inf'), 'mape': float('inf'), 'r2': -1}

def fast_walk_forward_validation(data: pd.DataFrame, target_column: str = 'total_load_actual') -> dict:
    """Run fast walk-forward validation for all models"""
    
    # Sample 30% of recent data for speed
    sample_size = int(len(data) * 0.3)
    data = data.tail(sample_size)
    
    n_splits = 3
    test_size = 12
    
    models = {
        'baseline': run_baseline_model,
        'arima': run_arima_model,
        'prophet': run_prophet_model,
        'lstm': run_lstm_model
    }
    
    results = {}
    
    for model_name, model_func in models.items():
        logger.info(f"Running {model_name}...")
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
                
                # Run model
                metrics = model_func(train_data, test_data, target_column)
                model_results.append(metrics)
                
            except Exception as e:
                logger.warning(f"{model_name} split {i} failed: {e}")
                model_results.append({'mae': float('inf'), 'mse': float('inf'), 'rmse': float('inf'), 'mape': float('inf'), 'r2': -1})
        
        results[model_name] = model_results
    
    return results

def summarize_results(results: dict) -> dict:
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

def main():
    """Main execution"""
    start_time = time.time()
    
    # Load data
    data_path = "data/processed/processed_energy_weather.csv"
    
    if not os.path.exists(data_path):
        logger.error(f"Data file not found: {data_path}")
        # Create sample data
        dates = pd.date_range('2020-01-01', periods=1000, freq='H')
        data = pd.DataFrame({
            'total_load_actual': 1000 + 200 * np.sin(np.arange(1000) * 2 * np.pi / 24) + np.random.normal(0, 50, 1000)
        }, index=dates)
    else:
        data = pd.read_csv(data_path, index_col=0, parse_dates=True)
    
    logger.info(f"Data shape: {data.shape}")
    
    # Check available packages
    available_packages = []
    
    try:
        import prophet
        available_packages.append("Prophet")
    except ImportError:
        pass
    
    try:
        import tensorflow
        available_packages.append("TensorFlow")
    except ImportError:
        pass
    
    logger.info(f"Available packages: {available_packages}")
    
    # Run validation
    results = fast_walk_forward_validation(data)
    
    # Summarize results
    summary = summarize_results(results)
    
    # Select best model
    best_model = None
    best_mae = float('inf')
    
    for model_name, metrics in summary.items():
        mae = metrics.get('mae_mean', float('inf'))
        if mae < best_mae:
            best_mae = mae
            best_model = model_name
    
    elapsed_time = time.time() - start_time
    
    # Print results
    print("\n" + "="*60)
    print("FAST WALK-FORWARD VALIDATION - ALL MODELS")
    print("="*60)
    print(f"Execution time: {elapsed_time:.2f} seconds")
    print(f"Available packages: {', '.join(available_packages) if available_packages else 'None'}")
    
    for model_name, metrics in summary.items():
        mae_mean = metrics.get('mae_mean', float('inf'))
        mae_std = metrics.get('mae_std', 0)
        rmse_mean = metrics.get('rmse_mean', float('inf'))
        rmse_std = metrics.get('rmse_std', 0)
        r2_mean = metrics.get('r2_mean', -1)
        r2_std = metrics.get('r2_std', 0)
        
        print(f"\n{model_name.upper()}:")
        if mae_mean != float('inf'):
            print(f"  MAE: {mae_mean:.2f} ± {mae_std:.2f}")
            print(f"  RMSE: {rmse_mean:.2f} ± {rmse_std:.2f}")
            print(f"  R²: {r2_mean:.4f} ± {r2_std:.4f}")
        else:
            print(f"  Status: FAILED/NOT AVAILABLE")
    
    print(f"\nBEST MODEL: {best_model}")
    print("="*60)

if __name__ == "__main__":
    main()
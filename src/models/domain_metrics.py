"""Domain-specific metrics for energy demand forecasting."""

import numpy as np
import pandas as pd
from typing import Dict

def calculate_peak_hour_accuracy(y_true: np.ndarray, y_pred: np.ndarray, 
                                 hours: np.ndarray, peak_hours: list = [8, 9, 17, 18, 19]) -> float:
    """Calculate accuracy during peak demand hours."""
    mask = np.isin(hours, peak_hours)
    if not mask.any():
        return np.nan
    return 1 - np.mean(np.abs(y_true[mask] - y_pred[mask]) / y_true[mask])

def calculate_mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean Absolute Percentage Error."""
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

def calculate_domain_metrics(y_true: np.ndarray, y_pred: np.ndarray, 
                            hours: np.ndarray = None) -> Dict[str, float]:
    """Calculate all domain-specific metrics."""
    metrics = {
        'mape': calculate_mape(y_true, y_pred),
        'max_error': np.max(np.abs(y_true - y_pred)),
        'peak_demand_error': np.abs(y_true.max() - y_pred.max())
    }
    
    if hours is not None:
        metrics['peak_hour_accuracy'] = calculate_peak_hour_accuracy(y_true, y_pred, hours)
    
    return metrics

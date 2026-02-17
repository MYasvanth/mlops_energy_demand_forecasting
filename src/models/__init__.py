"""
Models Package

This package contains model implementations for time series forecasting.

Modules:
- base: Base model interface (Strategy pattern)
- gbm_features: Feature engineering for gradient boosting models
- xgboost_model: XGBoost implementation
- lightgbm_model: LightGBM implementation
- train: Main training function

Usage:
    from src.models import train_multiple_models
    
    results = train_multiple_models(
        df,
        target_column='total_load_actual',
        models=['xgboost', 'lightgbm']
    )
"""

# Import main training function
from .train import train_multiple_models, create_model_comparison

# Import base classes
from .base import BaseTimeSeriesModel, ModelRegistry

# Import feature engineering
from .gbm_features import GBMFeatureEngineer, create_gbm_features

# Try to import GBM models (may fail if xgboost/lightgbm not installed)
try:
    from .xgboost_model import XGBoostTimeSeriesModel
    from .lightgbm_model import LightGBMTimeSeriesModel
    __all__ = [
        'train_multiple_models',
        'create_model_comparison',
        'BaseTimeSeriesModel',
        'ModelRegistry',
        'GBMFeatureEngineer',
        'create_gbm_features',
        'XGBoostTimeSeriesModel',
        'LightGBMTimeSeriesModel'
    ]
except ImportError:
    __all__ = [
        'train_multiple_models',
        'create_model_comparison',
        'BaseTimeSeriesModel',
        'ModelRegistry',
        'GBMFeatureEngineer',
        'create_gbm_features'
    ]

# Version
__version__ = '1.0.0'

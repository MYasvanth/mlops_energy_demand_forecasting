"""
Base Model Interface (Strategy Pattern)

This module defines the abstract base class for all time series forecasting models.
Implements the Strategy pattern to avoid giant if-else blocks and allow easy extension.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Tuple, Optional, List
import pandas as pd
import numpy as np
import mlflow
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import logging
import joblib
import os

logger = logging.getLogger(__name__)


class BaseTimeSeriesModel(ABC):
    """
    Abstract base class for time series forecasting models.
    Implements the Strategy pattern for model-agnostic training and evaluation.
    """
    
    def __init__(self, name: str, target_column: str = 'total_load_actual', 
                 n_splits: int = 5, random_state: int = 42):
        """
        Initialize the base model.
        
        Args:
            name: Model name for logging and identification
            target_column: Name of the target column
            n_splits: Number of splits for time-series cross-validation
            random_state: Random state for reproducibility
        """
        self.name = name
        self.target_column = target_column
        self.n_splits = n_splits
        self.random_state = random_state
        self.model = None
        self.best_params = None
        self.feature_names = []
        self.is_fitted = False
        
    @abstractmethod
    def fit(self, X_train: pd.DataFrame, y_train: pd.Series, 
            X_val: Optional[pd.DataFrame] = None, 
            y_val: Optional[pd.Series] = None) -> 'BaseTimeSeriesModel':
        """
        Train the model on the provided data.
        
        Args:
            X_train: Training features
            y_train: Training target
            X_val: Optional validation features
            y_val: Optional validation target
            
        Returns:
            Self for method chaining
        """
        pass
    
    @abstractmethod
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Make predictions on the provided data.
        
        Args:
            X: Features for prediction
            
        Returns:
            Array of predictions
        """
        pass
    
    @abstractmethod
    def get_feature_importance(self) -> Optional[Dict[str, float]]:
        """
        Get feature importance scores.
        
        Returns:
            Dictionary of feature names to importance scores, or None if not available
        """
        pass
    
    def time_series_cross_validate(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, Any]:
        """
        Perform time-series cross-validation with proper handling to avoid leakage.
        
        Uses TimeSeriesSplit to ensure no future data leaks into training.
        
        Args:
            X: Features
            y: Target
            
        Returns:
            Dictionary with CV metrics and fold information
        """
        logger.info(f"Performing time-series cross-validation with {self.n_splits} splits")
        
        tscv = TimeSeriesSplit(n_splits=self.n_splits)
        cv_scores = {
            'mae': [],
            'mse': [],
            'rmse': [],
            'r2': []
        }
        
        for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
            logger.info(f"CV Fold {fold + 1}/{self.n_splits}")
            
            X_train_fold = X.iloc[train_idx]
            y_train_fold = y.iloc[train_idx]
            X_val_fold = X.iloc[val_idx]
            y_val_fold = y.iloc[val_idx]
            
            # Fit model on this fold
            self.fit(X_train_fold, y_train_fold)
            
            # Predict on validation set
            y_pred = self.predict(X_val_fold)
            
            # Calculate metrics
            cv_scores['mae'].append(mean_absolute_error(y_val_fold, y_pred))
            cv_scores['mse'].append(mean_squared_error(y_val_fold, y_pred))
            cv_scores['rmse'].append(np.sqrt(mean_squared_error(y_val_fold, y_pred)))
            cv_scores['r2'].append(r2_score(y_val_fold, y_pred))
        
        # Calculate mean and std for each metric
        cv_results = {
            'mae_mean': np.mean(cv_scores['mae']),
            'mae_std': np.std(cv_scores['mae']),
            'mse_mean': np.mean(cv_scores['mse']),
            'mse_std': np.std(cv_scores['mse']),
            'rmse_mean': np.mean(cv_scores['rmse']),
            'rmse_std': np.std(cv_scores['rmse']),
            'r2_mean': np.mean(cv_scores['r2']),
            'r2_std': np.std(cv_scores['r2']),
            'fold_scores': cv_scores
        }
        
        logger.info(f"CV Results - MAE: {cv_results['mae_mean']:.4f} (+/- {cv_results['mae_std']:.4f})")
        logger.info(f"CV Results - R2: {cv_results['r2_mean']:.4f} (+/- {cv_results['r2_std']:.4f})")
        
        return cv_results
    
    def log_to_mlflow(self, run_id: Optional[str] = None) -> None:
        """
        Log model parameters, metrics, and artifacts to MLflow.
        
        Args:
            run_id: Optional MLflow run ID to log to
        """
        if run_id:
            with mlflow.start_run(run_id=run_id):
                self._log_params()
                self._log_metrics()
                self._log_artifacts()
                self._log_feature_importance()
        else:
            self._log_params()
            self._log_metrics()
            self._log_artifacts()
            self._log_feature_importance()
    
    def _log_params(self) -> None:
        """Log model parameters to MLflow."""
        mlflow.log_param("model_name", self.name)
        mlflow.log_param("target_column", self.target_column)
        mlflow.log_param("n_splits", self.n_splits)
        
        if self.best_params:
            for param, value in self.best_params.items():
                mlflow.log_param(f"best_{param}", value)
    
    def _log_metrics(self) -> None:
        """Log model metrics to MLflow."""
        if hasattr(self, 'train_metrics') and self.train_metrics:
            for metric, value in self.train_metrics.items():
                if np.isfinite(value):
                    mlflow.log_metric(f"train_{metric}", value)
        
        if hasattr(self, 'val_metrics') and self.val_metrics:
            for metric, value in self.val_metrics.items():
                if np.isfinite(value):
                    mlflow.log_metric(f"val_{metric}", value)
    
    @abstractmethod
    def _log_artifacts(self) -> None:
        """Log model artifacts to MLflow."""
        pass
    
    def _log_feature_importance(self) -> None:
        """Log feature importance to MLflow."""
        importance = self.get_feature_importance()
        if importance:
            # Log as JSON artifact
            importance_df = pd.DataFrame([
                {'feature': k, 'importance': v} 
                for k, v in importance.items()
            ])
            importance_df = importance_df.sort_values('importance', ascending=False)
            
            # Save to temporary file and log
            temp_path = f"{self.name}_feature_importance.csv"
            importance_df.to_csv(temp_path, index=False)
            mlflow.log_artifact(temp_path)
            os.remove(temp_path)
            
            # Also log top features as metrics
            top_features = importance_df.head(10)
            for idx, row in top_features.iterrows():
                mlflow.log_metric(f"importance_{row['feature']}", row['importance'])
    
    def save(self, path: str) -> None:
        """
        Save the model to disk.
        
        Args:
            path: Path to save the model
        """
        os.makedirs(os.path.dirname(path), exist_ok=True)
        model_data = {
            'model': self.model,
            'best_params': self.best_params,
            'feature_names': self.feature_names,
            'is_fitted': self.is_fitted,
            'name': self.name,
            'target_column': self.target_column
        }
        joblib.dump(model_data, path)
        logger.info(f"Model saved to {path}")
    
    def load(self, path: str) -> 'BaseTimeSeriesModel':
        """
        Load the model from disk.
        
        Args:
            path: Path to load the model from
            
        Returns:
            Self for method chaining
        """
        model_data = joblib.load(path)
        self.model = model_data['model']
        self.best_params = model_data['best_params']
        self.feature_names = model_data['feature_names']
        self.is_fitted = model_data['is_fitted']
        logger.info(f"Model loaded from {path}")
        return self
    
    def estimate_inference_latency(self, X_sample: pd.DataFrame, 
                                   n_iterations: int = 100) -> Dict[str, float]:
        """
        Estimate inference latency.
        
        Args:
            X_sample: Sample data for latency estimation
            n_iterations: Number of iterations for averaging
            
        Returns:
            Dictionary with latency statistics
        """
        import time
        
        # Warm-up run
        _ = self.predict(X_sample.head(1))
        
        # Timed runs
        latencies = []
        for _ in range(n_iterations):
            start = time.perf_counter()
            _ = self.predict(X_sample)
            end = time.perf_counter()
            latencies.append((end - start) * 1000)  # Convert to ms
        
        return {
            'mean_ms': np.mean(latencies),
            'std_ms': np.std(latencies),
            'min_ms': np.min(latencies),
            'max_ms': np.max(latencies),
            'p50_ms': np.percentile(latencies, 50),
            'p95_ms': np.percentile(latencies, 95),
            'p99_ms': np.percentile(latencies, 99)
        }


class ModelRegistry:
    """
    Registry for managing model strategies.
    Implements the Factory pattern for model creation.
    """
    
    _models: Dict[str, type] = {}
    
    @classmethod
    def register(cls, model_name: str, model_class: type) -> None:
        """
        Register a model class.
        
        Args:
            model_name: Name of the model
            model_class: Model class to register
        """
        cls._models[model_name] = model_class
        logger.info(f"Registered model: {model_name}")
    
    @classmethod
    def create(cls, model_name: str, **kwargs) -> BaseTimeSeriesModel:
        """
        Create a model instance.
        
        Args:
            model_name: Name of the model to create
            **kwargs: Arguments to pass to the model constructor
            
        Returns:
            Instance of the requested model
            
        Raises:
            ValueError: If model_name is not registered
        """
        if model_name not in cls._models:
            raise ValueError(f"Unknown model: {model_name}. Available: {list(cls._models.keys())}")
        
        return cls._models[model_name](**kwargs)
    
    @classmethod
    def available_models(cls) -> List[str]:
        """Get list of available model names."""
        return list(cls._models.keys())

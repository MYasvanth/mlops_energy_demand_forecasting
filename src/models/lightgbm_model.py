"""
LightGBM Model Implementation

This module implements LightGBM for time series forecasting using the Strategy pattern.
Features:
- Time-series cross-validation
- Optuna hyperparameter tuning
- Feature importance
- MLflow logging
- Zero leakage through proper feature engineering
- Fast inference for production deployment
"""

from typing import Dict, Any, Tuple, Optional
import pandas as pd
import numpy as np
import mlflow
import logging
import optuna
from optuna.samplers import TPESampler
import lightgbm as lgb
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from .base import BaseTimeSeriesModel, ModelRegistry
from .gbm_features import GBMFeatureEngineer

logger = logging.getLogger(__name__)


class LightGBMTimeSeriesModel(BaseTimeSeriesModel):
    """
    LightGBM model for time series forecasting.
    Implements the Strategy pattern with proper feature engineering.
    
    LightGBM is chosen for:
    - Fast training and inference (low latency)
    - Lower memory usage than XGBoost
    - Native categorical feature support
    - Often better accuracy on time series tasks
    """
    
    def __init__(self, 
                 target_column: str = 'total_load_actual',
                 n_splits: int = 5,
                 random_state: int = 42,
                 n_trials: int = 20,
                 lags: list = None,
                 rolling_windows: list = None):
        """
        Initialize LightGBM model.
        
        Args:
            target_column: Name of target column
            n_splits: Number of CV splits
            random_state: Random state
            n_trials: Number of Optuna trials for hyperparameter tuning
            lags: List of lag periods
            rolling_windows: List of rolling window sizes
        """
        super().__init__(
            name='lightgbm',
            target_column=target_column,
            n_splits=n_splits,
            random_state=random_state
        )
        
        self.n_trials = n_trials
        self.lags = lags or [1, 2, 3, 6, 12, 24, 48, 168]
        self.rolling_windows = rolling_windows or [3, 6, 12, 24, 48]
        
        # Feature engineer
        self.feature_engineer = GBMFeatureEngineer(
            target_column=target_column,
            lags=self.lags,
            rolling_windows=self.rolling_windows
        )
        
        # Training data storage for CV
        self.X_train = None
        self.y_train = None
        self.X_val = None
        self.y_val = None
        self.train_metrics = {}
        self.val_metrics = {}
        
    def _train_on_features(self, X_train: pd.DataFrame, y_train: pd.Series,
                          X_val: pd.DataFrame, y_val: pd.Series) -> None:
        """
        Train the model directly on already engineered features.
        This bypasses the internal feature engineering to avoid double processing.
        
        Args:
            X_train: Training features (already engineered)
            y_train: Training target
            X_val: Validation features (already engineered)
            y_val: Validation target
        """
        logger.info("Training LightGBM on pre-engineered features...")
        
        # Store data
        self.X_train = X_train
        self.y_train = y_train
        self.X_val = X_val
        self.y_val = y_val
        
        # Set feature names
        self.feature_names = list(X_train.columns)
        
        # Hyperparameter tuning with Optuna
        if self.n_trials > 0:
            self._tune_hyperparameters()
        
        # Train final model with best params
        self._train_final_model()
        
        # Calculate training metrics
        y_train_pred = self.model.predict(X_train)
        self.train_metrics = self._calculate_metrics(y_train, y_train_pred)
        
        # Calculate validation metrics
        if X_val is not None and len(X_val) > 0:
            y_val_pred = self.model.predict(X_val)
            self.val_metrics = self._calculate_metrics(y_val, y_val_pred)
        
        self.is_fitted = True
        logger.info(f"LightGBM training completed. Train MAE: {self.train_metrics.get('mae', float('inf')):.4f}")

    def fit(self, X_train: pd.DataFrame, y_train: pd.Series,
            X_val: Optional[pd.DataFrame] = None,
            y_val: Optional[pd.Series] = None) -> 'LightGBMTimeSeriesModel':
        """
        Fit the LightGBM model.
        
        Args:
            X_train: Training features (can be raw DataFrame or preprocessed)
            y_train: Training target
            X_val: Optional validation features
            y_val: Optional validation target
            
        Returns:
            Self for method chaining
        """
        logger.info("Fitting LightGBM model...")
        
        # Check if we need to do feature engineering
        if self.target_column in X_train.columns or isinstance(X_train, pd.DataFrame):
            # Prepare train/test split with feature engineering
            if isinstance(X_train, pd.DataFrame) and self.target_column in X_train.columns:
                df = X_train.copy()
            elif isinstance(X_train, pd.DataFrame):
                df = X_train.copy()
                df[self.target_column] = y_train
            else:
                raise ValueError("X_train must be a DataFrame with the target column")
            
            # Fit feature engineer on training data only (avoid leakage!)
            self.feature_engineer.fit(df)
            
            # Transform training data
            X_train_processed, y_train_processed = self.feature_engineer.transform(df)
            
            # Store feature names
            self.feature_names = self.feature_engineer.get_feature_names()
            
            # Handle validation data
            if X_val is not None and y_val is not None:
                if isinstance(X_val, pd.DataFrame) and self.target_column in X_val.columns:
                    df_val = X_val.copy()
                elif isinstance(X_val, pd.DataFrame):
                    df_val = X_val.copy()
                    df_val[self.target_column] = y_val
                else:
                    df_val = None
                    
                if df_val is not None:
                    X_val_processed, y_val_processed = self.feature_engineer.transform(df_val)
                else:
                    X_val_processed = X_val
                    y_val_processed = y_val
            else:
                # Create train/val split from training data
                val_size = int(len(X_train_processed) * 0.2)
                if val_size > 0:
                    X_val_processed = X_train_processed.iloc[-val_size:]
                    y_val_processed = y_train_processed.iloc[-val_size:]
                    X_train_processed = X_train_processed.iloc[:-val_size]
                    y_train_processed = y_train_processed.iloc[:-val_size]
                else:
                    X_val_processed = None
                    y_val_processed = None
            
            # Drop NaN rows
            train_mask = ~(X_train_processed.isna().any(axis=1) | y_train_processed.isna())
            X_train_processed = X_train_processed[train_mask]
            y_train_processed = y_train_processed[train_mask]
            
            if X_val_processed is not None:
                val_mask = ~(X_val_processed.isna().any(axis=1) | y_val_processed.isna())
                X_val_processed = X_val_processed[val_mask]
                y_val_processed = y_val_processed[val_mask]
                
        else:
            # Already preprocessed
            X_train_processed = X_train
            y_train_processed = y_train
            X_val_processed = X_val
            y_val_processed = y_val
            self.feature_names = list(X_train.columns)
        
        # Store training data
        self.X_train = X_train_processed
        self.y_train = y_train_processed
        self.X_val = X_val_processed
        self.y_val = y_val_processed
        
        # Hyperparameter tuning with Optuna
        if self.n_trials > 0:
            self._tune_hyperparameters()
        
        # Train final model with best params
        self._train_final_model()
        
        # Calculate training metrics
        y_train_pred = self.model.predict(X_train_processed)
        self.train_metrics = self._calculate_metrics(y_train_processed, y_train_pred)
        
        # Calculate validation metrics
        if X_val_processed is not None and len(X_val_processed) > 0:
            y_val_pred = self.model.predict(X_val_processed)
            self.val_metrics = self._calculate_metrics(y_val_processed, y_val_pred)
        
        self.is_fitted = True
        logger.info(f"LightGBM training completed. Train MAE: {self.train_metrics.get('mae', float('inf')):.4f}")
        
        return self
    
    def _tune_hyperparameters(self) -> None:
        """
        Tune hyperparameters using Optuna.
        """
        logger.info(f"Tuning LightGBM hyperparameters with {self.n_trials} trials...")
        
        def objective(trial):
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 50, 500),
                'max_depth': trial.suggest_int('max_depth', 3, 12),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                'num_leaves': trial.suggest_int('num_leaves', 20, 100),
                'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                'min_child_samples': trial.suggest_int('min_child_samples', 5, 50),
                'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
                'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
                'random_state': self.random_state,
                'n_jobs': -1,
                'verbose': -1
            }
            
            model = lgb.LGBMRegressor(**params)
            
            # Train on subset and validate
            if self.X_val is not None and len(self.X_val) > 0:
                model.fit(
                    self.X_train, self.y_train,
                    eval_set=[(self.X_val, self.y_val)],
                    callbacks=[lgb.early_stopping(50, verbose=False)]
                )
                y_pred = model.predict(self.X_val)
                mae = mean_absolute_error(self.y_val, y_pred)
                return mae
            else:
                # Use cross-validation
                from sklearn.model_selection import TimeSeriesSplit
                tscv = TimeSeriesSplit(n_splits=3)
                scores = []
                for train_idx, val_idx in tscv.split(self.X_train):
                    X_tr = self.X_train.iloc[train_idx]
                    y_tr = self.y_train.iloc[train_idx]
                    X_vl = self.X_train.iloc[val_idx]
                    y_vl = self.y_train.iloc[val_idx]
                    
                    model.fit(X_tr, y_tr, callbacks=[lgb.early_stopping(50, verbose=False)])
                    y_pred = model.predict(X_vl)
                    scores.append(mean_absolute_error(y_vl, y_pred))
                    
                return np.mean(scores)
        
        # Run optimization
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        sampler = TPESampler(seed=self.random_state)
        study = optuna.create_study(direction='minimize', sampler=sampler)
        study.optimize(objective, n_trials=self.n_trials, show_progress_bar=False)
        
        self.best_params = study.best_params
        logger.info(f"Best params: {self.best_params}")
        logger.info(f"Best CV score: {study.best_value:.4f}")
    
    def _train_final_model(self) -> None:
        """Train the final model with best parameters."""
        if self.best_params is None:
            # Default parameters
            self.best_params = {
                'n_estimators': 100,
                'max_depth': 6,
                'learning_rate': 0.1,
                'num_leaves': 31,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'random_state': self.random_state,
                'n_jobs': -1,
                'verbose': -1
            }
        
        # Add required parameters
        params = self.best_params.copy()
        params['random_state'] = self.random_state
        params['n_jobs'] = -1
        params['verbose'] = -1
        
        self.model = lgb.LGBMRegressor(**params)
        
        # Train
        if self.X_val is not None and len(self.X_val) > 0:
            self.model.fit(
                self.X_train, self.y_train,
                eval_set=[(self.X_val, self.y_val)],
                callbacks=[lgb.early_stopping(50, verbose=False)]
            )
        else:
            self.model.fit(self.X_train, self.y_train)
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Make predictions.
        
        Args:
            X: Features for prediction (can be raw or pre-engineered)
            
        Returns:
            Array of predictions
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")
        
        # Check if features are already engineered by comparing column names
        if hasattr(self, 'feature_names') and set(X.columns) == set(self.feature_names):
            # Features are already engineered, predict directly
            X_processed = X
        else:
            # Need to engineer features
            if self.target_column in X.columns or not isinstance(X, pd.DataFrame):
                if isinstance(X, pd.DataFrame) and self.target_column in X.columns:
                    df = X.copy()
                else:
                    df = X.copy()
                    df[self.target_column] = 0  # Placeholder
                    
                X_processed, _ = self.feature_engineer.transform(df)
                
                # Handle NaN
                X_processed = X_processed.fillna(0)
            else:
                X_processed = X
        
        return self.model.predict(X_processed)
    
    def get_feature_importance(self) -> Optional[Dict[str, float]]:
        """
        Get feature importance scores.
        
        Returns:
            Dictionary of feature names to importance scores
        """
        if not self.is_fitted or self.model is None:
            return None
            
        importance = self.model.feature_importances_
        return dict(zip(self.feature_names, importance.tolist()))
    
    def _calculate_metrics(self, y_true: pd.Series, y_pred: np.ndarray) -> Dict[str, float]:
        """Calculate evaluation metrics."""
        return {
            'mae': mean_absolute_error(y_true, y_pred),
            'mse': mean_squared_error(y_true, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
            'r2': r2_score(y_true, y_pred)
        }
    
    def _log_artifacts(self) -> None:
        """Log model artifacts to MLflow."""
        import os
        
        # Save model
        model_path = f"{self.name}_model.txt"
        self.model.booster_.save_model(model_path)
        mlflow.log_artifact(model_path)
        os.remove(model_path)
        
        # Save feature engineer
        fe_path = f"{self.name}_feature_engineer.joblib"
        self.feature_engineer.save(fe_path)
        mlflow.log_artifact(fe_path)
        os.remove(fe_path)
        
        # Log inference latency
        if self.X_train is not None and len(self.X_train) > 0:
            latency = self.estimate_inference_latency(self.X_train.head(100))
            mlflow.log_metric("inference_mean_ms", latency['mean_ms'])
            mlflow.log_metric("inference_p95_ms", latency['p95_ms'])
            mlflow.log_metric("inference_p99_ms", latency['p99_ms'])


# Register the model
ModelRegistry.register('lightgbm', LightGBMTimeSeriesModel)

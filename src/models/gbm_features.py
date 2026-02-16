"""
GBM Feature Engineering Module

This module handles feature engineering for gradient boosting models (XGBoost, LightGBM).
Separates feature engineering from training to ensure clean pipelines and avoid leakage.

Key features:
- Sliding window (lag) features
- Rolling statistics
- Time-based features
- Zero leakage through proper train/test splitting
"""

from typing import Tuple, List, Optional
import pandas as pd
import numpy as np
import logging
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)


class GBMFeatureEngineer:
    """
    Feature engineer for gradient boosting models.
    Creates lag features, rolling statistics, and time-based features.
    """
    
    def __init__(self, 
                 target_column: str = 'total_load_actual',
                 lags: List[int] = None,
                 rolling_windows: List[int] = None,
                 add_time_features: bool = True,
                 scaler: Optional[StandardScaler] = None):
        """
        Initialize the feature engineer.
        
        Args:
            target_column: Name of the target column
            lags: List of lag periods to create (e.g., [1, 2, 3, 6, 12, 24])
            rolling_windows: List of rolling window sizes (e.g., [3, 6, 12, 24])
            add_time_features: Whether to add time-based features
            scaler: Optional scaler for feature normalization
        """
        self.target_column = target_column
        self.lags = lags or [1, 2, 3, 6, 12, 24, 48, 168]  # hourly data: 1h, 2h, 3h, 6h, 12h, 24h, 48h, 1 week
        self.rolling_windows = rolling_windows or [3, 6, 12, 24, 48]
        self.add_time_features = add_time_features
        self.scaler = scaler or StandardScaler()
        self.feature_names = []
        self._is_fitted = False
        
    def fit(self, df: pd.DataFrame) -> 'GBMFeatureEngineer':
        """
        Fit the feature engineer (compute scaler parameters).
        
        Args:
            df: Input DataFrame
            
        Returns:
            Self for method chaining
        """
        logger.info("Fitting GBM feature engineer...")
        
        # Create features to get feature names
        features_df = self._create_features(df)
        
        # Exclude target and non-numeric columns
        exclude_cols = [self.target_column]
        if 'ds' in features_df.columns:
            exclude_cols.append('ds')
            
        self.feature_names = [col for col in features_df.columns 
                            if col not in exclude_cols and features_df[col].dtype in ['float64', 'int64']]
        
        # Fit scaler on training features only
        if len(self.feature_names) > 0:
            self.scaler.fit(features_df[self.feature_names].values)
            
        self._is_fitted = True
        logger.info(f"GBM feature engineer fitted with {len(self.feature_names)} features")
        
        return self
    
    def transform(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Transform the DataFrame to create features.
        
        Args:
            df: Input DataFrame
            
        Returns:
            Tuple of (features DataFrame, target Series)
        """
        if not self._is_fitted:
            raise ValueError("Feature engineer not fitted. Call fit() first.")
            
        # Create features
        features_df = self._create_features(df)
        
        # Extract target
        if self.target_column not in features_df.columns:
            raise ValueError(f"Target column {self.target_column} not found in DataFrame")
        
        target = features_df[self.target_column].copy()
        
        # Get feature columns (exclude target and non-numeric)
        exclude_cols = [self.target_column]
        if 'ds' in features_df.columns:
            exclude_cols.append('ds')
            
        feature_cols = [col for col in features_df.columns 
                      if col not in exclude_cols and features_df[col].dtype in ['float64', 'int64']]
        
        X = features_df[feature_cols]
        
        # Scale features
        X_scaled = pd.DataFrame(
            self.scaler.transform(X.values),
            columns=feature_cols,
            index=X.index
        )
        
        return X_scaled, target
    
    def fit_transform(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Fit and transform in one step.
        
        Args:
            df: Input DataFrame
            
        Returns:
            Tuple of (features DataFrame, target Series)
        """
        return self.fit(df).transform(df)
    
    def _create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create all features from the DataFrame.
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with all features
        """
        df = df.copy()
        
        # Add lag features (this is where zero leakage is critical!)
        df = self._add_lag_features(df)
        
        # Add rolling statistics
        df = self._add_rolling_features(df)
        
        # Add time-based features
        if self.add_time_features:
            df = self._add_time_features(df)
            
        return df
    
    def _add_lag_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add lag features for the target variable.
        CRITICAL: This must use past data only to avoid leakage!
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with lag features
        """
        if self.target_column not in df.columns:
            logger.warning(f"Target column {self.target_column} not found for lag features")
            return df
            
        target = df[self.target_column].copy()
        
        for lag in self.lags:
            df[f'lag_{lag}'] = target.shift(lag)
            
        return df
    
    def _add_rolling_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add rolling window statistics.
        CRITICAL: Must use shift(1) to avoid including current value!
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with rolling features
        """
        if self.target_column not in df.columns:
            return df
            
        target = df[self.target_column].copy()
        
        for window in self.rolling_windows:
            # Rolling mean (shifted by 1 to avoid leakage)
            df[f'rolling_mean_{window}'] = target.shift(1).rolling(window=window).mean()
            
            # Rolling std
            df[f'rolling_std_{window}'] = target.shift(1).rolling(window=window).std()
            
            # Rolling min
            df[f'rolling_min_{window}'] = target.shift(1).rolling(window=window).min()
            
            # Rolling max
            df[f'rolling_max_{window}'] = target.shift(1).rolling(window=window).max()
            
        return df
    
    def _add_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add time-based features.
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with time features
        """
        # Try to detect datetime column
        datetime_col = None
        
        # Check for common datetime column names
        for col in ['datetime', 'timestamp', 'ds', 'date', 'time']:
            if col in df.columns:
                if pd.api.types.is_datetime64_any_dtype(df[col]):
                    datetime_col = col
                    break
                    
        # If we have a datetime index, use it
        if df.index.name and 'datetime' in df.index.name.lower():
            datetime_col = df.index
            
        if datetime_col is None:
            # Try using the index if it's datetime
            if pd.api.types.is_datetime64_any_dtype(df.index):
                datetime_col = df.index
                
        if datetime_col is not None:
            if isinstance(datetime_col, pd.Index):
                dt = datetime_col
            else:
                dt = pd.to_datetime(df[datetime_col])
                
            # Extract time features
            df['hour'] = dt.hour
            df['day_of_week'] = dt.dayofweek
            df['day_of_month'] = dt.day
            df['month'] = dt.month
            df['quarter'] = dt.quarter
            df['is_weekend'] = (dt.dayofweek >= 5).astype(int)
            
            # Cyclical encoding for hour and day of week
            df['hour_sin'] = np.sin(2 * np.pi * dt.hour / 24)
            df['hour_cos'] = np.cos(2 * np.pi * dt.hour / 24)
            df['dow_sin'] = np.sin(2 * np.pi * dt.dayofweek / 7)
            df['dow_cos'] = np.cos(2 * np.pi * dt.dayofweek / 7)
            df['month_sin'] = np.sin(2 * np.pi * dt.month / 12)
            df['month_cos'] = np.cos(2 * np.pi * dt.month / 12)
            
        return df
    
    def prepare_train_test_split(self, df: pd.DataFrame, 
                                  train_size: float = 0.8) -> Tuple[Tuple[pd.DataFrame, pd.Series], Tuple[pd.DataFrame, pd.Series]]:
        """
        Prepare train/test split with proper feature engineering.
        CRITICAL: Fit on train only, transform both train and test!
        
        Args:
            df: Input DataFrame
            train_size: Proportion of data for training
            
        Returns:
            Tuple of ((X_train, y_train), (X_test, y_test))
        """
        # Calculate split point
        n = len(df)
        split_idx = int(n * train_size)
        
        # Split BEFORE feature engineering to avoid leakage!
        train_df = df.iloc[:split_idx].copy()
        test_df = df.iloc[split_idx:].copy()
        
        # Fit on training data ONLY
        self.fit(train_df)
        
        # Transform both train and test
        X_train, y_train = self.transform(train_df)
        X_test, y_test = self.transform(test_df)
        
        # Drop rows with NaN (from lag features at the beginning)
        # CRITICAL: This removes rows that can't be used due to lag requirements
        train_mask = ~(X_train.isna().any(axis=1) | y_train.isna())
        test_mask = ~(X_test.isna().any(axis=1) | y_test.isna())
        
        X_train = X_train[train_mask]
        y_train = y_train[train_mask]
        X_test = X_test[test_mask]
        y_test = y_test[test_mask]
        
        logger.info(f"Train size: {len(X_train)}, Test size: {len(X_test)}")
        logger.info(f"Number of features: {X_train.shape[1]}")
        
        return (X_train, y_train), (X_test, y_test)
    
    def get_feature_names(self) -> List[str]:
        """Get list of feature names."""
        return self.feature_names
    
    def save(self, path: str) -> None:
        """Save the feature engineer to disk."""
        import joblib
        import os
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump(self, path)
        logger.info(f"Feature engineer saved to {path}")
        
    @staticmethod
    def load(path: str) -> 'GBMFeatureEngineer':
        """Load the feature engineer from disk."""
        import joblib
        return joblib.load(path)


def create_gbm_features(df: pd.DataFrame, 
                        target_column: str = 'total_load_actual',
                        lags: List[int] = None,
                        rolling_windows: List[int] = None) -> Tuple[pd.DataFrame, pd.Series, GBMFeatureEngineer]:
    """
    Convenience function to create GBM features.
    
    Args:
        df: Input DataFrame
        target_column: Name of target column
        lags: List of lag periods
        rolling_windows: List of rolling window sizes
        
    Returns:
        Tuple of (X, y, feature_engineer)
    """
    engineer = GBMFeatureEngineer(
        target_column=target_column,
        lags=lags,
        rolling_windows=rolling_windows
    )
    
    X, y = engineer.fit_transform(df)
    
    return X, y, engineer

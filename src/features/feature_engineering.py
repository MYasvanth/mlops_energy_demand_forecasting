"""
Feature Engineering Module

This module creates advanced features for time series forecasting, including lag features,
rolling statistics, seasonal features, and domain-specific features for energy demand forecasting.
"""

import logging
from typing import Dict, List, Optional, Tuple
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_lag_features(df: pd.DataFrame, columns: List[str], lags: List[int]) -> pd.DataFrame:
    """
    Create lag features for specified columns with robust error handling.

    Args:
        df (pd.DataFrame): Input DataFrame with datetime index.
        columns (List[str]): Columns to create lags for.
        lags (List[int]): List of lag periods (in hours).

    Returns:
        pd.DataFrame: DataFrame with lag features added.
    """
    df_copy = df.copy()

    try:
        # Filter to only available columns
        available_columns = [col for col in columns if col in df_copy.columns]
        if not available_columns:
            logger.warning("No specified columns found for lag feature creation")
            return df_copy

        logger.info(f"Creating lag features for {len(available_columns)} columns with lags: {lags}")

        for col in available_columns:
            for lag in lags:
                try:
                    lag_feature = df_copy[col].shift(lag)
                    df_copy[f'{col}_lag_{lag}h'] = lag_feature
                except Exception as e:
                    logger.warning(f"Failed to create lag feature for {col} with lag {lag}: {e}")
                    continue

        # Remove rows with all NaN lag features (first max(lags) rows)
        if lags:
            max_lag = max(lags)
            original_shape = df_copy.shape[0]
            df_copy = df_copy.iloc[max_lag:].copy()
            logger.info(f"Removed {max_lag} rows with NaN lag features. Shape: {original_shape} -> {df_copy.shape[0]}")

        logger.info(f"Successfully created lag features. Final shape: {df_copy.shape}")
        return df_copy

    except Exception as e:
        logger.error(f"Error in lag feature creation: {str(e)}")
        return df  # Return original dataframe if feature creation fails


def create_rolling_features(df: pd.DataFrame, columns: List[str], windows: List[int],
                          functions: List[str] = ['mean', 'std', 'min', 'max']) -> pd.DataFrame:
    """
    Create rolling window features for specified columns.

    Args:
        df (pd.DataFrame): Input DataFrame with datetime index.
        columns (List[str]): Columns to create rolling features for.
        windows (List[int]): Rolling window sizes (in hours).
        functions (List[str]): Statistical functions to apply.

    Returns:
        pd.DataFrame: DataFrame with rolling features added.
    """
    df_copy = df.copy()
    for col in columns:
        for window in windows:
            for func in functions:
                if func == 'mean':
                    df_copy[f'{col}_rolling_{window}h_mean'] = df_copy[col].rolling(window=window).mean()
                elif func == 'std':
                    df_copy[f'{col}_rolling_{window}h_std'] = df_copy[col].rolling(window=window).std()
                elif func == 'min':
                    df_copy[f'{col}_rolling_{window}h_min'] = df_copy[col].rolling(window=window).min()
                elif func == 'max':
                    df_copy[f'{col}_rolling_{window}h_max'] = df_copy[col].rolling(window=window).max()

    logger.info(f"Created rolling features for {len(columns)} columns with {len(windows)} windows and {len(functions)} functions")
    return df_copy


def create_seasonal_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create seasonal features from datetime index.

    Args:
        df (pd.DataFrame): Input DataFrame with datetime index.

    Returns:
        pd.DataFrame: DataFrame with seasonal features added.
    """
    df_copy = df.copy()

    # Ensure datetime index
    if not isinstance(df_copy.index, pd.DatetimeIndex):
        raise ValueError("DataFrame must have datetime index for seasonal features")

    # Time-based features
    df_copy['hour'] = df_copy.index.hour
    df_copy['day_of_week'] = df_copy.index.dayofweek
    df_copy['month'] = df_copy.index.month
    df_copy['quarter'] = df_copy.index.quarter
    df_copy['day_of_year'] = df_copy.index.dayofyear
    df_copy['week_of_year'] = df_copy.index.isocalendar().week

    # Cyclic encoding for periodic features
    df_copy['hour_sin'] = np.sin(2 * np.pi * df_copy['hour'] / 24)
    df_copy['hour_cos'] = np.cos(2 * np.pi * df_copy['hour'] / 24)
    df_copy['month_sin'] = np.sin(2 * np.pi * df_copy['month'] / 12)
    df_copy['month_cos'] = np.cos(2 * np.pi * df_copy['month'] / 12)
    df_copy['day_of_week_sin'] = np.sin(2 * np.pi * df_copy['day_of_week'] / 7)
    df_copy['day_of_week_cos'] = np.cos(2 * np.pi * df_copy['day_of_week'] / 7)

    # Weekend indicator
    df_copy['is_weekend'] = df_copy['day_of_week'].isin([5, 6]).astype(int)

    # Business hours (simplified)
    df_copy['is_business_hours'] = ((df_copy['hour'] >= 9) & (df_copy['hour'] <= 17) &
                                   (df_copy['day_of_week'] < 5)).astype(int)

    logger.info("Created seasonal and time-based features")
    return df_copy


def create_weather_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create weather-specific features for energy forecasting.

    Args:
        df (pd.DataFrame): Input DataFrame with weather columns.

    Returns:
        pd.DataFrame: DataFrame with weather features added.
    """
    df_copy = df.copy()

    # Temperature features
    if 'temp' in df_copy.columns:
        df_copy['temp_category'] = pd.cut(df_copy['temp'],
                                        bins=[-np.inf, 0, 10, 20, 30, np.inf],
                                        labels=['freezing', 'cold', 'cool', 'warm', 'hot'])
        df_copy['temp_change'] = df_copy['temp'].diff()

    # Wind features
    if 'wind_speed' in df_copy.columns:
        df_copy['wind_category'] = pd.cut(df_copy['wind_speed'],
                                        bins=[0, 5, 10, 15, 20, np.inf],
                                        labels=['calm', 'light', 'moderate', 'strong', 'storm'])

    # Humidity categories
    if 'humidity' in df_copy.columns:
        df_copy['humidity_category'] = pd.cut(df_copy['humidity'],
                                            bins=[0, 30, 60, 80, 100],
                                            labels=['dry', 'comfortable', 'humid', 'very_humid'])

    # Weather condition encoding (if available)
    if 'weather_main' in df_copy.columns:
        weather_dummies = pd.get_dummies(df_copy['weather_main'], prefix='weather')
        df_copy = pd.concat([df_copy, weather_dummies], axis=1)

    logger.info("Created weather-specific features")
    return df_copy


def create_energy_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create energy-specific features for demand forecasting.

    Args:
        df (pd.DataFrame): Input DataFrame with energy columns.

    Returns:
        pd.DataFrame: DataFrame with energy features added.
    """
    df_copy = df.copy()

    # Load factor (actual load vs forecast)
    if 'total_load_actual' in df_copy.columns and 'total_load_forecast' in df_copy.columns:
        df_copy['load_factor'] = df_copy['total_load_actual'] / df_copy['total_load_forecast']
        df_copy['load_error'] = df_copy['total_load_actual'] - df_copy['total_load_forecast']

    # Renewable energy ratio
    renewable_cols = ['generation_solar', 'generation_wind_onshore', 'generation_hydro_run_of_river_and_poundage',
                     'generation_hydro_water_reservoir', 'generation_other_renewable']
    fossil_cols = ['generation_fossil_coal_derived_gas', 'generation_fossil_gas', 'generation_fossil_hard_coal',
                  'generation_fossil_oil', 'generation_fossil_brown_coal_lignite']

    available_renewable = [col for col in renewable_cols if col in df_copy.columns]
    available_fossil = [col for col in fossil_cols if col in df_copy.columns]

    if available_renewable:
        df_copy['total_renewable'] = df_copy[available_renewable].sum(axis=1)
    if available_fossil:
        df_copy['total_fossil'] = df_copy[available_fossil].sum(axis=1)

    if 'total_renewable' in df_copy.columns and 'total_fossil' in df_copy.columns:
        total_generation = df_copy['total_renewable'] + df_copy['total_fossil']
        # Avoid division by zero
        df_copy['renewable_ratio'] = df_copy['total_renewable'] / total_generation.replace(0, np.nan)

    # Price volatility
    if 'price_actual' in df_copy.columns:
        df_copy['price_volatility'] = df_copy['price_actual'].rolling(window=24).std()

    logger.info("Created energy-specific features")
    return df_copy


def create_interaction_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create interaction features between different variables.

    Args:
        df (pd.DataFrame): Input DataFrame.

    Returns:
        pd.DataFrame: DataFrame with interaction features added.
    """
    df_copy = df.copy()

    # Weather and time interactions
    if 'temp' in df_copy.columns and 'hour' in df_copy.columns:
        df_copy['temp_hour_interaction'] = df_copy['temp'] * df_copy['hour_sin']

    if 'wind_speed' in df_copy.columns and 'generation_wind_onshore' in df_copy.columns:
        df_copy['wind_generation_interaction'] = df_copy['wind_speed'] * df_copy['generation_wind_onshore']

    # Load and weather interactions
    if 'total_load_actual' in df_copy.columns and 'temp' in df_copy.columns:
        df_copy['load_temp_interaction'] = df_copy['total_load_actual'] * df_copy['temp']

    logger.info("Created interaction features")
    return df_copy


def create_target_features(df: pd.DataFrame, target_column: str, horizons: List[int]) -> pd.DataFrame:
    """
    Create target features for multi-step forecasting.

    Args:
        df (pd.DataFrame): Input DataFrame.
        target_column (str): Name of the target column.
        horizons (List[int]): Forecast horizons (in hours).

    Returns:
        pd.DataFrame: DataFrame with target features added.
    """
    df_copy = df.copy()
    for horizon in horizons:
        df_copy[f'{target_column}_target_{horizon}h'] = df_copy[target_column].shift(-horizon)

    logger.info(f"Created target features for horizons: {horizons}")
    return df_copy


def full_feature_engineering_pipeline(df: pd.DataFrame, target_column: str = 'total_load_actual',
                                    lag_periods: List[int] = [1, 2, 6, 12, 24, 48, 72],
                                    rolling_windows: List[int] = [6, 12, 24, 48],
                                    forecast_horizons: List[int] = [1, 6, 12, 24]) -> pd.DataFrame:
    """
    Run complete feature engineering pipeline.

    Args:
        df (pd.DataFrame): Input DataFrame with datetime index.
        target_column (str): Target column for forecasting.
        lag_periods (List[int]): Lag periods for lag features.
        rolling_windows (List[int]): Rolling window sizes.
        forecast_horizons (List[int]): Forecast horizons for targets.

    Returns:
        pd.DataFrame: DataFrame with all engineered features.
    """
    logger.info("Starting full feature engineering pipeline...")

    # Seasonal features
    df = create_seasonal_features(df)

    # Lag features for key columns
    key_columns = [target_column, 'generation_solar', 'generation_wind_onshore', 'temp', 'wind_speed']
    available_key_cols = [col for col in key_columns if col in df.columns]
    df = create_lag_features(df, available_key_cols, lag_periods)

    # Rolling features
    df = create_rolling_features(df, available_key_cols, rolling_windows)

    # Domain-specific features
    df = create_weather_features(df)
    df = create_energy_features(df)

    # Interaction features
    df = create_interaction_features(df)

    # Target features for forecasting
    df = create_target_features(df, target_column, forecast_horizons)

    logger.info("Full feature engineering pipeline completed")
    return df


if __name__ == "__main__":
    # Example usage
    import sys
    from pathlib import Path

    sys.path.append(str(Path(__file__).parent.parent / 'src'))
    from data.preprocessing import full_preprocessing_pipeline
    from data.ingestion import ingest_data

    # Load and preprocess data
    energy_path = Path(__file__).parent.parent / 'data' / 'raw' / 'energy_dataset.csv'
    weather_path = Path(__file__).parent.parent / 'data' / 'raw' / 'weather_features.csv'

    raw_data = ingest_data(str(energy_path), str(weather_path))
    processed_data = full_preprocessing_pipeline(raw_data['energy'], raw_data['weather'])

    # Feature engineering
    feature_data = full_feature_engineering_pipeline(processed_data)

    print("Feature engineering completed successfully.")
    print(f"Feature engineered data shape: {feature_data.shape}")
    print(f"Number of features: {len(feature_data.columns)}")

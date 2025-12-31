"""
Data Preprocessing Module

This module handles data cleaning, normalization, and preprocessing for energy demand forecasting.
It includes handling missing values, outlier detection, normalization, and time series specific preprocessing.
"""

import logging
from typing import Dict, List, Optional, Tuple, Union
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.impute import SimpleImputer
import warnings
from src.utils.exceptions import DataValidationError

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')


def validate_data_schema(df: pd.DataFrame, schema: Dict[str, str]) -> None:
    """
    Validate DataFrame schema against expected column types.

    Args:
        df (pd.DataFrame): DataFrame to validate.
        schema (Dict[str, str]): Expected schema as column_name: dtype.

    Raises:
        DataValidationError: If schema validation fails.
    """
    for col, expected_dtype in schema.items():
        if col not in df.columns:
            raise DataValidationError(f"Missing column: {col}", field=col)

        actual_dtype = str(df[col].dtype)
        if actual_dtype != expected_dtype:
            raise DataValidationError(
                f"Column {col} has dtype {actual_dtype}, expected {expected_dtype}",
                field=col, value=actual_dtype
            )

    logger.info("Data schema validation passed")


def handle_missing_values(df: pd.DataFrame, strategy: str = 'mean', columns: Optional[List[str]] = None) -> pd.DataFrame:
    """
    Handle missing values in the DataFrame using specified strategy with robust error handling.

    Args:
        df (pd.DataFrame): Input DataFrame.
        strategy (str): Imputation strategy ('mean', 'median', 'most_frequent', 'constant').
        columns (Optional[List[str]]): Columns to impute. If None, impute all numeric columns.

    Returns:
        pd.DataFrame: DataFrame with missing values handled.
    """
    df_copy = df.copy()

    try:
        if columns is None:
            columns = df_copy.select_dtypes(include=[np.number]).columns.tolist()

        logger.info(f"Handling missing values using {strategy} strategy for columns: {columns}")

        if not columns:
            logger.warning("No numeric columns found for imputation")
            return df_copy

        # Validate strategy
        valid_strategies = ['mean', 'median', 'most_frequent', 'constant']
        if strategy not in valid_strategies:
            logger.warning(f"Invalid strategy '{strategy}', using 'mean'")
            strategy = 'mean'

        imputer = SimpleImputer(strategy=strategy)
        df_copy[columns] = imputer.fit_transform(df_copy[columns])

        remaining_nans = df_copy.isnull().sum().sum()
        logger.info(f"Missing values handled. Remaining NaNs: {remaining_nans}")

        return df_copy

    except Exception as e:
        logger.error(f"Error in missing value handling: {str(e)}")
        # Fallback: use forward fill then backward fill
        try:
            logger.info("Attempting fallback missing value handling...")
            df_copy[columns] = df_copy[columns].fillna(method='ffill').fillna(method='bfill')
            # Fill any remaining NaNs with column means
            for col in columns:
                if df_copy[col].isnull().any():
                    col_mean = df_copy[col].mean()
                    if pd.isna(col_mean):
                        col_mean = 0  # Final fallback
                    df_copy[col] = df_copy[col].fillna(col_mean)
            logger.info("Fallback missing value handling completed")
            return df_copy
        except Exception as fallback_e:
            logger.error(f"Fallback missing value handling also failed: {str(fallback_e)}")
            # Return original dataframe if all else fails
            return df


def detect_outliers_iqr(df: pd.DataFrame, column: str, factor: float = 1.5) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Detect outliers using IQR method.

    Args:
        df (pd.DataFrame): Input DataFrame.
        column (str): Column to check for outliers.
        factor (float): IQR factor for outlier detection.

    Returns:
        Tuple[pd.DataFrame, pd.Series]: DataFrame with outliers removed and boolean mask of outliers.
    """
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - factor * IQR
    upper_bound = Q3 + factor * IQR

    outliers = (df[column] < lower_bound) | (df[column] > upper_bound)
    df_clean = df[~outliers].copy()

    logger.info(f"Detected {outliers.sum()} outliers in {column} using IQR method")
    return df_clean, outliers


def remove_outliers(df: pd.DataFrame, columns: List[str], method: str = 'iqr', **kwargs) -> pd.DataFrame:
    """
    Remove outliers from specified columns.

    Args:
        df (pd.DataFrame): Input DataFrame.
        columns (List[str]): Columns to remove outliers from.
        method (str): Outlier detection method ('iqr').
        **kwargs: Additional arguments for outlier detection method.

    Returns:
        pd.DataFrame: DataFrame with outliers removed.
    """
    df_copy = df.copy()
    for col in columns:
        if method == 'iqr':
            df_copy, _ = detect_outliers_iqr(df_copy, col, **kwargs)

    logger.info(f"Outliers removed from columns: {columns}")
    return df_copy


def normalize_data(df: pd.DataFrame, columns: List[str], method: str = 'standard') -> Tuple[pd.DataFrame, dict]:
    """
    Normalize specified columns using the chosen method.

    Args:
        df (pd.DataFrame): Input DataFrame.
        columns (List[str]): Columns to normalize.
        method (str): Normalization method ('standard' or 'minmax').

    Returns:
        Tuple[pd.DataFrame, dict]: Normalized DataFrame and scalers dictionary.
    """
    df_copy = df.copy()
    scalers = {}

    for col in columns:
        if method == 'standard':
            scaler = StandardScaler()
        elif method == 'minmax':
            scaler = MinMaxScaler()
        else:
            raise ValueError(f"Unknown normalization method: {method}")

        df_copy[col] = scaler.fit_transform(df_copy[[col]])
        scalers[col] = scaler

    logger.info(f"Normalized columns {columns} using {method} scaling")
    return df_copy, scalers


def parse_timestamps(df: pd.DataFrame, time_column: str, format: Optional[str] = None) -> pd.DataFrame:
    """
    Parse timestamp column and set it as index.

    Args:
        df (pd.DataFrame): Input DataFrame.
        time_column (str): Name of the timestamp column.
        format (Optional[str]): Datetime format string.

    Returns:
        pd.DataFrame: DataFrame with parsed timestamps as index.
    """
    df_copy = df.copy()
    df_copy[time_column] = pd.to_datetime(df_copy[time_column], format=format)
    df_copy.set_index(time_column, inplace=True)
    df_copy.sort_index(inplace=True)

    logger.info(f"Parsed timestamps and set {time_column} as index")
    return df_copy


def resample_time_series(df: pd.DataFrame, freq: str, method: str = 'mean') -> pd.DataFrame:
    """
    Resample time series data to a different frequency.

    Args:
        df (pd.DataFrame): Time series DataFrame with datetime index.
        freq (str): Resampling frequency (e.g., 'H' for hourly, 'D' for daily).
        method (str): Resampling method ('mean', 'sum', 'first', 'last').

    Returns:
        pd.DataFrame: Resampled DataFrame.
    """
    if method == 'mean':
        df_resampled = df.resample(freq).mean()
    elif method == 'sum':
        df_resampled = df.resample(freq).sum()
    elif method == 'first':
        df_resampled = df.resample(freq).first()
    elif method == 'last':
        df_resampled = df.resample(freq).last()
    else:
        raise ValueError(f"Unknown resampling method: {method}")

    logger.info(f"Resampled data to {freq} frequency using {method} method")
    return df_resampled


def create_lag_features(df: pd.DataFrame, columns: List[str], lags: List[int]) -> pd.DataFrame:
    """
    Create lag features for time series data.

    Args:
        df (pd.DataFrame): Time series DataFrame.
        columns (List[str]): Columns to create lags for.
        lags (List[int]): List of lag periods.

    Returns:
        pd.DataFrame: DataFrame with lag features added.
    """
    df_copy = df.copy()
    for col in columns:
        for lag in lags:
            df_copy[f'{col}_lag_{lag}'] = df_copy[col].shift(lag)

    logger.info(f"Created lag features for columns {columns} with lags {lags}")
    return df_copy


def preprocess_energy_data(energy_df: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocess energy dataset with cleaning, missing value handling, and feature engineering.

    Args:
        energy_df (pd.DataFrame): Raw energy DataFrame.

    Returns:
        pd.DataFrame: Preprocessed energy DataFrame.
    """
    logger.info("Preprocessing energy data...")

    # Parse timestamps
    energy_df = parse_timestamps(energy_df, 'time')

    # Handle missing values
    numeric_cols = energy_df.select_dtypes(include=[np.number]).columns.tolist()
    energy_df = handle_missing_values(energy_df, strategy='mean', columns=numeric_cols)

    # Remove outliers from key columns
    key_cols = ['total_load_actual']
    energy_df = remove_outliers(energy_df, key_cols, method='iqr')

    # Create lag features for forecasting
    lag_cols = ['total_load_actual']
    energy_df = create_lag_features(energy_df, lag_cols, lags=[1, 24, 168])  # 1h, 1d, 1w

    logger.info("Energy data preprocessing completed")
    return energy_df


def preprocess_weather_data(weather_df: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocess weather dataset with cleaning and feature engineering.

    Args:
        weather_df (pd.DataFrame): Raw weather DataFrame.

    Returns:
        pd.DataFrame: Preprocessed weather DataFrame.
    """
    logger.info("Preprocessing weather data...")

    # Parse timestamps
    weather_df = parse_timestamps(weather_df, 'dt_iso')

    # Handle missing values
    numeric_cols = weather_df.select_dtypes(include=[np.number]).columns.tolist()
    weather_df = handle_missing_values(weather_df, strategy='mean', columns=numeric_cols)

    # Resample to hourly if needed (assuming data is already hourly)
    # weather_df = resample_time_series(weather_df, 'H')

    logger.info("Weather data preprocessing completed")
    return weather_df


def merge_datasets(energy_df: pd.DataFrame, weather_df: pd.DataFrame) -> pd.DataFrame:
    """
    Merge energy and weather datasets on timestamp.

    Args:
        energy_df (pd.DataFrame): Preprocessed energy DataFrame.
        weather_df (pd.DataFrame): Preprocessed weather DataFrame.

    Returns:
        pd.DataFrame: Merged DataFrame.
    """
    logger.info("Merging energy and weather datasets...")

    # Ensure both have datetime index
    merged_df = pd.merge(energy_df, weather_df, left_index=True, right_index=True, how='inner')

    # Ensure the index remains datetime and handle timezone issues
    if not isinstance(merged_df.index, pd.DatetimeIndex):
        try:
            merged_df.index = pd.to_datetime(merged_df.index, utc=True)
        except ValueError:
            merged_df.index = pd.to_datetime(merged_df.index)
        merged_df.sort_index(inplace=True)

    logger.info(f"Merged dataset shape: {merged_df.shape}")
    return merged_df


def preprocess_data(merged_df: pd.DataFrame, config: Dict) -> pd.DataFrame:
    """
    Preprocess the merged dataset according to configuration.

    Args:
        merged_df (pd.DataFrame): Merged energy and weather DataFrame.
        config (Dict): Configuration dictionary.

    Returns:
        pd.DataFrame: Preprocessed DataFrame.
    """
    logger.info("Preprocessing merged data...")

    # Parse timestamps if not already
    if 'time' in merged_df.columns:
        merged_df = parse_timestamps(merged_df, 'time')

    # Handle missing values
    numeric_cols = merged_df.select_dtypes(include=[np.number]).columns.tolist()
    merged_df = handle_missing_values(merged_df, strategy='mean', columns=numeric_cols)

    # Remove outliers from key columns
    key_cols = config.get('outlier_columns', ['total_load_actual', 'price_actual'])
    merged_df = remove_outliers(merged_df, key_cols, method='iqr')

    # Normalize if specified
    if config.get('normalize', False):
        norm_cols = config.get('normalize_columns', ['total_load_actual'])
        merged_df, _ = normalize_data(merged_df, norm_cols, method=config.get('normalize_method', 'standard'))

    logger.info("Data preprocessing completed")
    return merged_df


def full_preprocessing_pipeline(energy_df: pd.DataFrame, weather_df: pd.DataFrame) -> pd.DataFrame:
    """
    Run full preprocessing pipeline.

    Args:
        energy_df (pd.DataFrame): Raw energy DataFrame.
        weather_df (pd.DataFrame): Raw weather DataFrame.

    Returns:
        pd.DataFrame: Fully preprocessed and merged DataFrame.
    """
    logger.info("Starting full preprocessing pipeline...")

    energy_processed = preprocess_energy_data(energy_df)
    weather_processed = preprocess_weather_data(weather_df)
    merged_df = merge_datasets(energy_processed, weather_processed)

    logger.info("Full preprocessing pipeline completed")
    return merged_df


if __name__ == "__main__":
    # Example usage
    try:
        from src.data.ingestion import ingest_data

        energy_path = "data/raw/energy_dataset.csv"
        weather_path = "data/raw/weather_features.csv"
        data = ingest_data(energy_path, weather_path)

        processed_data = full_preprocessing_pipeline(data['energy'], data['weather'])
        print("Preprocessing completed successfully.")
        print(f"Processed data shape: {processed_data.shape}")
    except Exception as e:
        print(f"Error in preprocessing example: {e}")
        import traceback
        traceback.print_exc()

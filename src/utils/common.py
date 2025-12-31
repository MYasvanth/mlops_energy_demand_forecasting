"""Common utilities shared across all modules."""

import logging
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, Union
from pathlib import Path


def setup_logging(name: str, level: int = logging.INFO) -> logging.Logger:
    """Centralized logging setup."""
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger(name)


def load_and_parse_data(file_path: str, time_column: str = 'time') -> pd.DataFrame:
    """Centralized data loading with timestamp parsing."""
    df = pd.read_csv(file_path)
    if time_column in df.columns:
        df[time_column] = pd.to_datetime(df[time_column])
        df.set_index(time_column, inplace=True)
    return df


def validate_dataframe(df: pd.DataFrame, required_columns: list = None) -> None:
    """Centralized DataFrame validation."""
    if df.empty:
        raise ValueError("DataFrame is empty")
    
    if required_columns:
        missing_cols = set(required_columns) - set(df.columns)
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")


def safe_operation(operation, fallback_value=None, logger=None):
    """Centralized error handling wrapper."""
    def decorator(func):
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                if logger:
                    logger.error(f"{operation} failed: {e}")
                return fallback_value
        return wrapper
    return decorator
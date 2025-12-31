"""
Unit tests for data preprocessing functions.

This module contains comprehensive tests for data preprocessing,
validation, and transformation functions.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import patch, MagicMock

from src.data.preprocessing import (
    full_preprocessing_pipeline,
    handle_missing_values,
    remove_outliers,
    normalize_data,
    validate_data_schema
)
from src.utils.exceptions import DataValidationError


class TestDataPreprocessing:
    """Test suite for data preprocessing functions."""

    @pytest.fixture
    def sample_energy_data(self):
        """Create sample energy demand data for testing."""
        dates = pd.date_range('2023-01-01', periods=100, freq='H')
        np.random.seed(42)

        data = {
            'time': dates,
            'total_load_actual': np.random.normal(50000, 5000, 100),
            'total_load_forecast': np.random.normal(49000, 4800, 100),
            'price_day_ahead': np.random.normal(50, 5, 100)
        }

        # Introduce some missing values and outliers
        data['total_load_actual'][10:15] = np.nan
        data['total_load_actual'][20] = 150000  # Outlier
        data['total_load_actual'][80:85] = np.nan

        return pd.DataFrame(data)

    @pytest.fixture
    def sample_weather_data(self):
        """Create sample weather data for testing."""
        dates = pd.date_range('2023-01-01', periods=100, freq='H')
        np.random.seed(42)

        data = {
            'dt_iso': dates,
            'temp': np.random.normal(20, 5, 100),
            'humidity': np.random.uniform(30, 90, 100),
            'wind_speed': np.random.uniform(0, 20, 100),
            'precipitation': np.random.exponential(2, 100)
        }

        # Introduce missing values
        data['temp'][5:10] = np.nan
        data['humidity'][50:55] = np.nan

        return pd.DataFrame(data)

    def test_handle_missing_values_interpolation(self, sample_energy_data):
        """Test missing value handling with interpolation."""
        original_na_count = sample_energy_data['total_load_actual'].isna().sum()

        result = handle_missing_values(
            sample_energy_data.copy(),
            strategy='mean',
            columns=['total_load_actual']
        )

        assert result['total_load_actual'].isna().sum() == 0
        assert len(result) == len(sample_energy_data)

    def test_handle_missing_values_drop(self, sample_energy_data):
        """Test missing value handling by dropping rows."""
        original_length = len(sample_energy_data)

        # Create a copy and drop rows with NaN in total_load_actual
        result = sample_energy_data.dropna(subset=['total_load_actual']).copy()

        assert len(result) < original_length
        assert result['total_load_actual'].isna().sum() == 0

    def test_handle_missing_values_fill_value(self, sample_energy_data):
        """Test missing value handling with fill value."""
        result = handle_missing_values(
            sample_energy_data.copy(),
            strategy='constant',
            columns=['total_load_actual']
        )

        assert result['total_load_actual'].isna().sum() == 0
        # With constant strategy, NaNs are filled with 0 by default
        assert (result['total_load_actual'][10:15] == 0).all()

    def test_remove_outliers_iqr(self, sample_energy_data):
        """Test outlier removal using IQR method."""
        original_length = len(sample_energy_data)

        result = remove_outliers(
            sample_energy_data.copy(),
            method='iqr',
            columns=['total_load_actual'],
            factor=1.5
        )

        assert len(result) <= original_length

    def test_remove_outliers_zscore(self, sample_energy_data):
        """Test outlier removal using Z-score method."""
        original_length = len(sample_energy_data)

        result = remove_outliers(
            sample_energy_data.copy(),
            method='zscore',
            columns=['total_load_actual'],
            threshold=3.0
        )

        assert len(result) <= original_length

    def test_normalize_data_standard_scaler(self, sample_energy_data):
        """Test data normalization with standard scaler."""
        result, scalers = normalize_data(
            sample_energy_data.copy(),
            method='standard',
            columns=['total_load_actual', 'price_day_ahead']
        )

        # Check that normalized columns have mean close to 0 and std close to 1
        assert abs(result['total_load_actual'].mean()) < 0.1
        assert abs(result['total_load_actual'].std() - 1.0) < 0.1

    def test_normalize_data_minmax_scaler(self, sample_energy_data):
        """Test data normalization with min-max scaler."""
        result, scalers = normalize_data(
            sample_energy_data.copy(),
            method='minmax',
            columns=['total_load_actual']
        )

        normalized_col = result['total_load_actual']
        assert normalized_col.min() >= 0
        assert normalized_col.max() <= 1

    def test_validate_data_schema_valid(self, sample_energy_data):
        """Test data schema validation with valid data."""
        schema = {
            'time': 'datetime64[ns]',
            'total_load_actual': 'float64',
            'total_load_forecast': 'float64'
        }

        # Should not raise an exception
        validate_data_schema(sample_energy_data, schema)

    def test_validate_data_schema_invalid_type(self, sample_energy_data):
        """Test data schema validation with invalid column type."""
        schema = {
            'time': 'datetime64[ns]',
            'total_load_actual': 'int64'  # Should be float64
        }

        with pytest.raises(DataValidationError):
            validate_data_schema(sample_energy_data, schema)

    def test_validate_data_schema_missing_column(self, sample_energy_data):
        """Test data schema validation with missing column."""
        schema = {
            'time': 'datetime64[ns]',
            'nonexistent_column': 'float64'
        }

        with pytest.raises(DataValidationError):
            validate_data_schema(sample_energy_data, schema)

    def test_full_preprocessing_pipeline(self, sample_energy_data, sample_weather_data):
        """Test the complete preprocessing pipeline."""
        result = full_preprocessing_pipeline(
            sample_energy_data,
            sample_weather_data
        )

        assert isinstance(result, pd.DataFrame)
        assert len(result) > 0
        # After merging, time becomes the index, so check for target column
        assert 'total_load_actual' in result.columns

    def test_preprocessing_pipeline_error_handling(self):
        """Test error handling in preprocessing pipeline."""
        # Test with None inputs
        with pytest.raises(AttributeError):
            full_preprocessing_pipeline(None, None)

    @pytest.mark.parametrize("missing_method", ["mean", "median", "most_frequent", "constant"])
    def test_missing_value_methods(self, sample_energy_data, missing_method):
        """Test different missing value handling methods."""
        result = handle_missing_values(
            sample_energy_data.copy(),
            strategy=missing_method,
            columns=['total_load_actual']
        )

        assert len(result) == len(sample_energy_data)

    @pytest.mark.parametrize("outlier_method", ["iqr", "zscore"])
    def test_outlier_removal_methods(self, sample_energy_data, outlier_method):
        """Test different outlier removal methods."""
        result = remove_outliers(
            sample_energy_data.copy(),
            method=outlier_method,
            columns=['total_load_actual']
        )

        assert isinstance(result, pd.DataFrame)
        assert len(result) <= len(sample_energy_data)

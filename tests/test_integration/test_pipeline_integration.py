"""
Integration tests for the complete ML pipeline.

This module contains tests that verify the integration between
different components of the energy demand forecasting pipeline.
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock
from pathlib import Path
import tempfile
import os

from src.data.ingestion import ingest_data
from src.data.preprocessing import full_preprocessing_pipeline
from src.features.feature_engineering import full_feature_engineering_pipeline
from src.models.train import train_multiple_models
from src.models.predict import create_prediction_pipeline
from src.monitoring.evidently_monitoring import EvidentlyMonitor
from src.utils.exceptions import DataValidationError, ModelTrainingError


class TestPipelineIntegration:
    """Test suite for complete pipeline integration."""

    @pytest.fixture
    def sample_raw_data(self, tmp_path):
        """Create sample raw data files for testing."""
        # Create temporary data files
        energy_data = {
            'time': pd.date_range('2023-01-01', periods=200, freq='H'),
            'total_load_actual': np.random.normal(50000, 5000, 200),
            'total_load_forecast': np.random.normal(49000, 4800, 200),
            'price_day_ahead': np.random.normal(50, 5, 200)
        }

        weather_data = {
            'time': pd.date_range('2023-01-01', periods=200, freq='H'),
            'temp': np.random.normal(20, 5, 200),
            'humidity': np.random.uniform(30, 90, 200),
            'wind_speed': np.random.uniform(0, 20, 200)
        }

        energy_df = pd.DataFrame(energy_data)
        weather_df = pd.DataFrame(weather_data)

        # Save to temporary files
        energy_path = tmp_path / "energy_dataset.csv"
        weather_path = tmp_path / "weather_features.csv"

        energy_df.to_csv(energy_path, index=False)
        weather_df.to_csv(weather_path, index=False)

        return str(energy_path), str(weather_path)

    def test_data_ingestion_to_preprocessing_integration(self, sample_raw_data):
        """Test integration from data ingestion to preprocessing."""
        energy_path, weather_path = sample_raw_data

        # Test data ingestion
        raw_data = ingest_data(energy_path, weather_path)

        assert 'energy' in raw_data
        assert 'weather' in raw_data
        assert isinstance(raw_data['energy'], pd.DataFrame)
        assert isinstance(raw_data['weather'], pd.DataFrame)

        # Test preprocessing
        processed_data = full_preprocessing_pipeline(
            raw_data['energy'],
            raw_data['weather']
        )

        assert isinstance(processed_data, pd.DataFrame)
        assert len(processed_data) > 0
        assert 'total_load_actual' in processed_data.columns

    def test_preprocessing_to_feature_engineering_integration(self, sample_raw_data):
        """Test integration from preprocessing to feature engineering."""
        energy_path, weather_path = sample_raw_data

        # Get processed data
        raw_data = ingest_data(energy_path, weather_path)
        processed_data = full_preprocessing_pipeline(
            raw_data['energy'],
            raw_data['weather']
        )

        # Test feature engineering
        feature_data = full_feature_engineering_pipeline(processed_data)

        assert isinstance(feature_data, pd.DataFrame)
        assert len(feature_data) > 0

        # Should have additional feature columns
        original_cols = set(processed_data.columns)
        feature_cols = set(feature_data.columns)
        new_cols = feature_cols - original_cols

        assert len(new_cols) > 0  # Should have added new features

    def test_feature_engineering_to_model_training_integration(self, sample_raw_data):
        """Test integration from feature engineering to model training."""
        energy_path, weather_path = sample_raw_data

        # Get feature data
        raw_data = ingest_data(energy_path, weather_path)
        processed_data = full_preprocessing_pipeline(
            raw_data['energy'],
            raw_data['weather']
        )
        feature_data = full_feature_engineering_pipeline(processed_data)

        # Test model training
        with patch('src.models.train.TimeSeriesTrainer') as mock_trainer_class:
            mock_trainer = MagicMock()
            mock_trainer.evaluate_model.return_value = {
                'mae': 1000, 'rmse': 1200, 'r2': 0.8, 'mape': 2.0
            }
            mock_trainer.model_type = 'arima'
            mock_trainer_class.return_value = mock_trainer

            results = train_multiple_models(
                feature_data,
                target_column='total_load_actual',
                models=['arima']
            )

            assert 'arima' in results
            assert 'metrics' in results['arima']
            assert 'model' in results['arima']

    def test_model_training_to_prediction_integration(self, sample_raw_data, tmp_path):
        """Test integration from model training to prediction."""
        energy_path, weather_path = sample_raw_data

        # Get feature data
        raw_data = ingest_data(energy_path, weather_path)
        processed_data = full_preprocessing_pipeline(
            raw_data['energy'],
            raw_data['weather']
        )
        feature_data = full_feature_engineering_pipeline(processed_data)

        # Mock training and create prediction pipeline
        with patch('src.models.train.TimeSeriesTrainer') as mock_trainer_class:
            mock_trainer = MagicMock()
            mock_trainer.evaluate_model.return_value = {
                'mae': 1000, 'rmse': 1200, 'r2': 0.8, 'mape': 2.0
            }
            mock_trainer.model_type = 'arima'
            mock_trainer_class.return_value = mock_trainer

            # Train model
            results = train_multiple_models(
                feature_data,
                target_column='total_load_actual',
                models=['arima']
            )

            # Create prediction pipeline
            model_path = tmp_path / "test_model"
            predictor = create_prediction_pipeline('arima', str(model_path))

            # Test prediction
            test_data = feature_data.tail(24)  # Last 24 hours
            predictions = predictor.predict(test_data, steps=6)

            assert isinstance(predictions, np.ndarray)
            assert len(predictions) == 6

    def test_monitoring_integration(self, sample_raw_data):
        """Test monitoring integration with the pipeline."""
        energy_path, weather_path = sample_raw_data

        # Get processed data
        raw_data = ingest_data(energy_path, weather_path)
        processed_data = full_preprocessing_pipeline(
            raw_data['energy'],
            raw_data['weather']
        )

        # Create monitor
        monitor = EvidentlyMonitor(processed_data, target_column='total_load_actual')

        # Test monitoring cycle
        with patch('src.monitoring.evidently_monitoring.Report') as mock_report:
            mock_report_instance = MagicMock()
            mock_report.return_value = mock_report_instance
            mock_report_instance.run.return_value = None

            # Run monitoring cycle
            results = monitor.run_monitoring_cycle(
                processed_data.tail(50),
                np.random.normal(50000, 1000, 10)
            )

            assert isinstance(results, dict)
            # Should have monitoring results even if mocked

    def test_error_propagation_through_pipeline(self, sample_raw_data):
        """Test error propagation through the pipeline."""
        energy_path, weather_path = sample_raw_data

        # Test with invalid data
        invalid_data = pd.DataFrame({'wrong_column': [1, 2, 3]})

        # Should raise error in preprocessing
        with pytest.raises((DataValidationError, KeyError)):
            full_preprocessing_pipeline(invalid_data, invalid_data)

    def test_pipeline_with_missing_data(self, tmp_path):
        """Test pipeline behavior with missing data files."""
        # Test with non-existent files
        with pytest.raises(FileNotFoundError):
            ingest_data("nonexistent_energy.csv", "nonexistent_weather.csv")

    @pytest.mark.slow
    def test_full_pipeline_performance(self, sample_raw_data):
        """Test full pipeline performance (marked as slow)."""
        import time
        energy_path, weather_path = sample_raw_data

        start_time = time.time()

        # Run full pipeline
        raw_data = ingest_data(energy_path, weather_path)
        processed_data = full_preprocessing_pipeline(
            raw_data['energy'],
            raw_data['weather']
        )
        feature_data = full_feature_engineering_pipeline(processed_data)

        with patch('src.models.train.TimeSeriesTrainer') as mock_trainer_class:
            mock_trainer = MagicMock()
            mock_trainer.evaluate_model.return_value = {
                'mae': 1000, 'rmse': 1200, 'r2': 0.8, 'mape': 2.0
            }
            mock_trainer.model_type = 'arima'
            mock_trainer_class.return_value = mock_trainer

            results = train_multiple_models(
                feature_data,
                target_column='total_load_actual',
                models=['arima']
            )

        end_time = time.time()
        duration = end_time - start_time

        # Pipeline should complete within reasonable time (adjust threshold as needed)
        assert duration < 30  # Less than 30 seconds for this test data size

    def test_data_consistency_through_pipeline(self, sample_raw_data):
        """Test data consistency throughout the pipeline."""
        energy_path, weather_path = sample_raw_data

        # Run through pipeline
        raw_data = ingest_data(energy_path, weather_path)
        processed_data = full_preprocessing_pipeline(
            raw_data['energy'],
            raw_data['weather']
        )
        feature_data = full_feature_engineering_pipeline(processed_data)

        # Check that essential columns are preserved
        assert 'total_load_actual' in feature_data.columns

        # Check that time index is maintained (if applicable)
        if 'time' in feature_data.columns:
            assert pd.api.types.is_datetime64_any_dtype(feature_data['time'])

        # Check for data leakage (future data in past predictions)
        # This is a basic check - more sophisticated validation would be needed
        assert len(feature_data) > 0

    def test_configuration_consistency(self):
        """Test that configurations are consistent across components."""
        # This would test that model configs, data schemas, etc. are consistent
        # For now, just check that imports work
        from src.data.preprocessing import full_preprocessing_pipeline
        from src.features.feature_engineering import full_feature_engineering_pipeline
        from src.models.train import train_multiple_models

        assert callable(full_preprocessing_pipeline)
        assert callable(full_feature_engineering_pipeline)
        assert callable(train_multiple_models)

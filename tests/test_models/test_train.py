"""
Unit tests for model training functions.

This module contains comprehensive tests for model training,
hyperparameter optimization, and model evaluation functions.
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock, mock_open
from pathlib import Path
import tempfile
import os

from src.models.train import (
    TimeSeriesTrainer,
    train_multiple_models,
    create_model_comparison
)
from src.utils.exceptions import ModelTrainingError, DataValidationError


class TestTimeSeriesTrainer:
    """Test suite for TimeSeriesTrainer class."""

    @pytest.fixture
    def sample_training_data(self):
        """Create sample training data for testing."""
        dates = pd.date_range('2023-01-01', periods=200, freq='H')
        np.random.seed(42)

        data = {
            'time': dates,
            'total_load_actual': np.random.normal(50000, 5000, 200),
            'temp': np.random.normal(20, 5, 200),
            'humidity': np.random.uniform(30, 90, 200)
        }

        df = pd.DataFrame(data)
        df.set_index('time', inplace=True)
        return df

    @pytest.fixture
    def trainer(self):
        """Create a TimeSeriesTrainer instance."""
        return TimeSeriesTrainer(model_type='arima', target_column='total_load_actual')

    def test_trainer_initialization(self):
        """Test trainer initialization."""
        trainer = TimeSeriesTrainer(model_type='arima', target_column='total_load_actual')

        assert trainer.model_type == 'arima'
        assert trainer.target_column == 'total_load_actual'
        assert trainer.model is None
        assert trainer.best_params is None

    def test_prepare_data_arima(self, trainer, sample_training_data):
        """Test data preparation for ARIMA model."""
        train, test = trainer.prepare_data_arima(sample_training_data, 'total_load_actual')

        assert len(train) == 160  # 80% of 200
        assert len(test) == 40    # 20% of 200
        assert isinstance(train, pd.Series)
        assert isinstance(test, pd.Series)

    def test_prepare_data_prophet(self, trainer, sample_training_data):
        """Test data preparation for Prophet model."""
        # Reset index for Prophet format
        data = sample_training_data.reset_index()

        train, test = trainer.prepare_data_prophet(data, 'total_load_actual')

        assert 'ds' in train.columns
        assert 'y' in train.columns
        assert len(train) == 160
        assert len(test) == 40

    def test_train_arima_model(self, trainer, sample_training_data):
        """Test ARIMA model training."""
        with patch('statsmodels.tsa.arima.model.ARIMA.fit') as mock_fit:
            mock_model = MagicMock()
            mock_fit.return_value = mock_model

            trainer.train_arima(sample_training_data)

            assert trainer.model is not None
            assert trainer.best_params is not None
            mock_fit.assert_called_once()

    def test_train_prophet_model(self, trainer, sample_training_data):
        """Test Prophet model training."""
        with patch('prophet.Prophet.fit') as mock_fit:
            mock_model = MagicMock()
            mock_fit.return_value = mock_model

            trainer.train_prophet(sample_training_data)

            assert trainer.model is not None
            mock_fit.assert_called_once()

    def test_train_lstm_model(self, trainer, sample_training_data):
        """Test LSTM model training."""
        with patch('tensorflow.keras.models.Sequential.fit') as mock_fit:
            mock_model = MagicMock()
            mock_fit.return_value = mock_model

            trainer.train_lstm(sample_training_data, epochs=1, batch_size=32)

            assert trainer.model is not None
            mock_fit.assert_called_once()

    def test_evaluate_model_arima(self, trainer, sample_training_data):
        """Test model evaluation for ARIMA."""
        # Train first
        with patch('statsmodels.tsa.arima.model.ARIMA.fit') as mock_fit:
            mock_model = MagicMock()
            mock_model.forecast.return_value = np.array([50000, 51000, 52000])
            mock_fit.return_value = mock_model

            trainer.train_arima(sample_training_data)

            # Evaluate
            metrics = trainer.evaluate_model(sample_training_data)

            assert 'mae' in metrics
            assert 'rmse' in metrics
            assert 'r2' in metrics
            assert 'mape' in metrics

    def test_save_and_load_model(self, trainer, sample_training_data, tmp_path):
        """Test model saving and loading."""
        # Train a model first
        with patch('statsmodels.tsa.arima.model.ARIMA.fit') as mock_fit:
            mock_model = MagicMock()
            mock_fit.return_value = mock_model

            trainer.train_arima(sample_training_data)

            # Save model
            model_path = tmp_path / "test_model"
            trainer.save_model(str(model_path))

            # Create new trainer and load model
            new_trainer = TimeSeriesTrainer(model_type='arima', target_column='total_load_actual')
            new_trainer.load_model(str(model_path))

            assert new_trainer.best_params is not None

    def test_invalid_model_type(self):
        """Test invalid model type handling."""
        with pytest.raises(ValueError):
            TimeSeriesTrainer(model_type='invalid_type')

    def test_train_with_invalid_data(self, trainer):
        """Test training with invalid data."""
        invalid_data = pd.DataFrame({'wrong_column': [1, 2, 3]})

        with pytest.raises((KeyError, DataValidationError)):
            trainer.train_arima(invalid_data)


class TestModelTrainingFunctions:
    """Test suite for model training utility functions."""

    @pytest.fixture
    def sample_data(self):
        """Create sample data for testing."""
        dates = pd.date_range('2023-01-01', periods=100, freq='H')
        np.random.seed(42)

        data = {
            'time': dates,
            'total_load_actual': np.random.normal(50000, 5000, 100),
            'temp': np.random.normal(20, 5, 100)
        }

        df = pd.DataFrame(data)
        df.set_index('time', inplace=True)
        return df

    @patch('src.models.train.TimeSeriesTrainer')
    def test_train_multiple_models(self, mock_trainer_class, sample_data):
        """Test training multiple models."""
        # Mock trainer instance
        mock_trainer = MagicMock()
        mock_trainer.evaluate_model.return_value = {
            'mae': 1000, 'rmse': 1200, 'r2': 0.8, 'mape': 2.0
        }
        mock_trainer.model_type = 'arima'
        mock_trainer_class.return_value = mock_trainer

        results = train_multiple_models(
            sample_data,
            target_column='total_load_actual',
            models=['arima', 'prophet']
        )

        assert 'arima' in results
        assert 'prophet' in results
        assert 'metrics' in results['arima']
        assert 'model' in results['arima']

    def test_create_model_comparison(self):
        """Test model comparison creation."""
        mock_results = {
            'arima': {
                'metrics': {'mae': 1000, 'rmse': 1200, 'r2': 0.8},
                'model': MagicMock()
            },
            'prophet': {
                'metrics': {'mae': 1200, 'rmse': 1400, 'r2': 0.7},
                'model': MagicMock()
            },
            'lstm': {
                'metrics': {'mae': 800, 'rmse': 1000, 'r2': 0.9},
                'model': MagicMock()
            }
        }

        comparison = create_model_comparison(mock_results)

        assert 'model_comparison' in comparison
        assert 'best_model' in comparison
        assert 'best_mae' in comparison
        assert comparison['best_model'] == 'lstm'
        assert comparison['best_mae'] == 800

    @patch('mlflow.log_metric')
    @patch('mlflow.log_param')
    @patch('mlflow.set_experiment')
    def test_mlflow_integration(self, mock_set_experiment, mock_log_param, mock_log_metric, sample_data):
        """Test MLflow integration in training."""
        with patch('src.models.train.TimeSeriesTrainer') as mock_trainer_class:
            mock_trainer = MagicMock()
            mock_trainer.evaluate_model.return_value = {
                'mae': 1000, 'rmse': 1200, 'r2': 0.8, 'mape': 2.0
            }
            mock_trainer.model_type = 'arima'
            mock_trainer_class.return_value = mock_trainer

            train_multiple_models(
                sample_data,
                target_column='total_load_actual',
                models=['arima'],
                experiment_name='test_experiment'
            )

            mock_set_experiment.assert_called_with('test_experiment')
            assert mock_log_metric.called or mock_log_param.called

    def test_error_handling_invalid_model(self, sample_data):
        """Test error handling for invalid model types."""
        with pytest.raises(ValueError):
            train_multiple_models(
                sample_data,
                target_column='total_load_actual',
                models=['invalid_model']
            )

    def test_empty_data_handling(self):
        """Test handling of empty data."""
        empty_data = pd.DataFrame()

        with pytest.raises((DataValidationError, ValueError)):
            train_multiple_models(
                empty_data,
                target_column='total_load_actual',
                models=['arima']
            )

    @pytest.mark.parametrize("model_type", ["arima", "prophet", "lstm"])
    def test_different_model_types(self, model_type, sample_data):
        """Test training with different model types."""
        with patch('src.models.train.TimeSeriesTrainer') as mock_trainer_class:
            mock_trainer = MagicMock()
            mock_trainer.evaluate_model.return_value = {
                'mae': 1000, 'rmse': 1200, 'r2': 0.8, 'mape': 2.0
            }
            mock_trainer.model_type = model_type
            mock_trainer_class.return_value = mock_trainer

            results = train_multiple_models(
                sample_data,
                target_column='total_load_actual',
                models=[model_type]
            )

            assert model_type in results
            assert 'metrics' in results[model_type]

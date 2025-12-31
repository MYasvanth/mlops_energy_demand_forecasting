"""
Integration Tests for Energy Demand Forecasting MLOps Pipeline

This module contains comprehensive integration tests that validate
the end-to-end functionality of the ML pipeline including ZenML,
MLflow, Optuna, Evidently, and Prefect integrations.
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
from unittest.mock import patch, MagicMock
import tempfile
import shutil
from datetime import datetime, timedelta

# Import project modules
from src.data.ingestion import ingest_data
from src.data.preprocessing import full_preprocessing_pipeline
from src.features.feature_engineering import full_feature_engineering_pipeline
from src.models.train import train_multiple_models
from src.models.predict import TimeSeriesPredictor, EnsemblePredictor
from src.monitoring.evidently_monitoring import EvidentlyMonitor, MonitoringPipeline
from zenml_pipelines.training_pipeline import energy_demand_training_pipeline
from prefect_flows.orchestration_flow import energy_demand_orchestration_flow


class TestEndToEndPipeline:
    """Test the complete ML pipeline from data to predictions."""

    @pytest.fixture
    def sample_data(self, tmp_path):
        """Create sample energy and weather data for testing."""
        # Create sample energy data
        dates = pd.date_range('2023-01-01', periods=1000, freq='h')
        np.random.seed(42)

        energy_data = pd.DataFrame({
            'time': dates,
            'total_load_actual': 1000 + np.random.normal(0, 100, 1000) + 200 * np.sin(2 * np.pi * np.arange(1000) / 24),
            'total_load_forecast': 1000 + np.random.normal(0, 80, 1000) + 180 * np.sin(2 * np.pi * np.arange(1000) / 24),
            'generation_solar': np.maximum(0, 300 * np.sin(2 * np.pi * np.arange(1000) / 24)),
            'generation_wind_onshore': np.maximum(0, 200 + np.random.normal(0, 50, 1000)),
            'price_actual': 50 + np.random.normal(0, 10, 1000)
        })

        # Create sample weather data
        weather_data = pd.DataFrame({
            'dt_iso': dates,
            'temp': 15 + 10 * np.sin(2 * np.pi * np.arange(1000) / (24 * 365)) + np.random.normal(0, 5, 1000),
            'humidity': 60 + np.random.normal(0, 20, 1000),
            'wind_speed': np.maximum(0, 5 + np.random.normal(0, 3, 1000)),
            'pressure': 1013 + np.random.normal(0, 10, 1000)
        })

        # Save to temporary files
        energy_path = tmp_path / "energy_dataset.csv"
        weather_path = tmp_path / "weather_features.csv"

        energy_data.to_csv(energy_path, index=False)
        weather_data.to_csv(weather_path, index=False)

        return str(energy_path), str(weather_path)

    def test_data_ingestion_pipeline(self, sample_data):
        """Test data ingestion and validation."""
        energy_path, weather_path = sample_data

        # Test data ingestion
        data = ingest_data(energy_path, weather_path)

        assert 'energy' in data
        assert 'weather' in data
        assert len(data['energy']) > 0
        assert len(data['weather']) > 0

        # Check required columns exist
        required_energy_cols = ['time', 'total_load_actual']
        required_weather_cols = ['dt_iso', 'temp']

        for col in required_energy_cols:
            assert col in data['energy'].columns

        for col in required_weather_cols:
            assert col in data['weather'].columns

    def test_data_preprocessing_pipeline(self, sample_data):
        """Test data preprocessing pipeline."""
        energy_path, weather_path = sample_data

        # Load and preprocess data
        raw_data = ingest_data(energy_path, weather_path)
        processed_data = full_preprocessing_pipeline(raw_data['energy'], raw_data['weather'])

        # Check preprocessing results
        assert isinstance(processed_data, pd.DataFrame)
        assert len(processed_data) > 0
        assert 'time' in processed_data.index.names or 'time' in processed_data.columns

        # Check for engineered features (note: lag features are created with different naming)
        # The full preprocessing pipeline only does basic preprocessing, not feature engineering
        # Feature engineering is done separately in full_feature_engineering_pipeline
        assert 'total_load_actual_lag_24' in processed_data.columns
        # hour and day_of_week are added in feature engineering, not preprocessing
        # So we check for basic preprocessing features instead
        assert 'total_load_actual' in processed_data.columns
        assert 'temp' in processed_data.columns

    def test_feature_engineering_pipeline(self, sample_data):
        """Test feature engineering pipeline."""
        energy_path, weather_path = sample_data

        # Load and process data
        raw_data = ingest_data(energy_path, weather_path)
        processed_data = full_preprocessing_pipeline(raw_data['energy'], raw_data['weather'])
        feature_data = full_feature_engineering_pipeline(processed_data)

        # Check feature engineering results
        assert len(feature_data.columns) > len(processed_data.columns)

        # Check for seasonal features
        seasonal_features = ['hour_sin', 'hour_cos', 'month_sin', 'month_cos']
        for feature in seasonal_features:
            assert feature in feature_data.columns

        # Check for target features
        target_features = ['total_load_actual_target_1h', 'total_load_actual_target_24h']
        for feature in target_features:
            assert feature in feature_data.columns

    @patch('mlflow.start_run')
    @patch('mlflow.log_param')
    @patch('mlflow.log_metric')
    def test_model_training_pipeline(self, mock_log_metric, mock_log_param, mock_start_run, sample_data):
        """Test model training with MLflow integration."""
        energy_path, weather_path = sample_data

        # Load and process data
        raw_data = ingest_data(energy_path, weather_path)
        processed_data = full_preprocessing_pipeline(raw_data['energy'], raw_data['weather'])
        feature_data = full_feature_engineering_pipeline(processed_data)

        # Mock MLflow
        mock_run = MagicMock()
        mock_start_run.return_value.__enter__ = MagicMock(return_value=mock_run)
        mock_start_run.return_value.__exit__ = MagicMock(return_value=None)

        # Train models
        results = train_multiple_models(feature_data, models=['arima', 'prophet'])

        # Check results
        assert 'arima' in results
        assert 'prophet' in results

        for model_name, result in results.items():
            if model_name in ['arima', 'prophet', 'lstm']:
                assert 'metrics' in result
                assert 'mae' in result['metrics']
                assert 'r2' in result['metrics']
                assert result['metrics']['mae'] > 0
            else:
                # For registry and other keys, just check they exist
                assert isinstance(result, (dict, str, float))

    def test_prediction_pipeline(self, sample_data):
        """Test prediction pipeline."""
        energy_path, weather_path = sample_data

        # Load and process data
        raw_data = ingest_data(energy_path, weather_path)
        processed_data = full_preprocessing_pipeline(raw_data['energy'], raw_data['weather'])
        feature_data = full_feature_engineering_pipeline(processed_data)

        # Train a simple model for testing
        results = train_multiple_models(feature_data, models=['arima'])

        # Test prediction
        predictor = TimeSeriesPredictor('arima')
        # Load the trained model
        predictor.load_model('models/arima_model')

        test_data = feature_data.tail(50)
        predictions = predictor.predict(test_data, steps=24)

        assert len(predictions) == 24
        assert all(isinstance(pred, (int, float, np.number)) for pred in predictions)

    def test_evidently_monitoring(self, sample_data):
        """Test Evidently monitoring integration."""
        energy_path, weather_path = sample_data

        # Load reference data
        raw_data = ingest_data(energy_path, weather_path)
        reference_data = full_preprocessing_pipeline(raw_data['energy'], raw_data['weather'])

        # Initialize monitor
        monitor = EvidentlyMonitor(reference_data)

        # Test data drift detection
        current_data = reference_data.copy()
        # Introduce slight drift
        current_data['total_load_actual'] = current_data['total_load_actual'] * 1.05

        drift_results = monitor.detect_data_drift(current_data)

        assert 'drift_detected' in drift_results
        assert 'drift_score' in drift_results
        assert isinstance(drift_results['drift_detected'], bool)

    def test_monitoring_pipeline(self, sample_data, tmp_path):
        """Test complete monitoring pipeline."""
        energy_path, weather_path = sample_data

        # Setup monitoring
        monitor_pipeline = MonitoringPipeline(energy_path)
        monitor_pipeline.initialize_monitoring()

        # Load test data
        raw_data = ingest_data(energy_path, weather_path)
        current_data = full_preprocessing_pipeline(raw_data['energy'], raw_data['weather'])

        # Run monitoring cycle
        results = monitor_pipeline.run_monitoring_cycle(
            current_data,
            output_dir=str(tmp_path / "monitoring")
        )

        # Check results structure
        assert 'data_quality' in results
        assert 'data_drift' in results
        assert 'model_performance' in results
        assert 'alerts' in results
        assert isinstance(results['alerts'], list)

    @patch('zenml_pipelines.training_pipeline.energy_demand_training_pipeline')
    def test_zenml_pipeline_integration(self, mock_pipeline, sample_data):
        """Test ZenML pipeline integration."""
        energy_path, weather_path = sample_data

        # Mock ZenML pipeline
        mock_pipeline.return_value = None

        # Call pipeline (would normally run ZenML)
        energy_demand_training_pipeline(energy_path, weather_path)

        # Verify pipeline was called
        mock_pipeline.assert_called_once_with(energy_path, weather_path, 'total_load_actual')

    @patch('prefect.flow')
    @patch('src.data.ingestion.ingest_data')
    def test_prefect_orchestration(self, mock_ingest, mock_flow_decorator, sample_data):
        """Test Prefect orchestration flow."""
        energy_path, weather_path = sample_data

        # Mock dependencies
        mock_ingest.return_value = {'energy': pd.DataFrame(), 'weather': pd.DataFrame()}

        # Mock flow decorator
        mock_flow = MagicMock()
        mock_flow_decorator.return_value = mock_flow

        # This would normally run the Prefect flow
        # In testing, we just verify the structure exists
        assert callable(energy_demand_orchestration_flow)

    def test_ensemble_prediction(self, sample_data):
        """Test ensemble prediction functionality."""
        energy_path, weather_path = sample_data

        # Load data
        raw_data = ingest_data(energy_path, weather_path)
        processed_data = full_preprocessing_pipeline(raw_data['energy'], raw_data['weather'])
        feature_data = full_feature_engineering_pipeline(processed_data)

        # Create mock model configs
        model_configs = [
            {
                'model_type': 'arima',
                'model_path': 'models/test_arima_model',
                'weight': 0.5
            },
            {
                'model_type': 'prophet',
                'model_path': 'models/test_prophet_model',
                'weight': 0.5
            }
        ]

        # Test ensemble initialization (would fail without actual models)
        with pytest.raises(Exception):
            ensemble = EnsemblePredictor(model_configs)

    def test_error_handling(self, sample_data):
        """Test error handling in pipeline components."""
        # Test with invalid file paths
        with pytest.raises(FileNotFoundError):
            ingest_data("nonexistent_energy.csv", "nonexistent_weather.csv")

        # Test with invalid model type
        predictor = TimeSeriesPredictor('invalid_model')
        with pytest.raises(ValueError):
            predictor.predict(pd.DataFrame(), steps=24)


class TestAPIFunctionality:
    """Test API functionality (requires FastAPI test client)."""

    def test_api_endpoints_structure(self):
        """Test that API endpoints are properly defined."""
        from src.deployment.fastapi_app import app

        # Check that main routes exist
        routes = [route.path for route in app.routes]

        assert "/" in routes
        assert "/health" in routes
        assert "/predict" in routes
        assert "/models" in routes
        assert "/ensemble-predict" in routes


class TestConfigurationValidation:
    """Test configuration validation and loading."""

    def test_model_config_loading(self):
        """Test loading model configuration."""
        config_path = "configs/model/model_config.yaml"

        if Path(config_path).exists():
            import yaml
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)

            # Check required sections
            assert 'model' in config
            assert 'target_column' in config['model']
            assert 'models' in config
            assert 'arima' in config['models']
            assert 'prophet' in config['models']
            assert 'lstm' in config['models']

    def test_requirements_satisfaction(self):
        """Test that requirements.txt contains necessary packages."""
        requirements_path = "requirements.txt"

        if Path(requirements_path).exists():
            with open(requirements_path, 'r') as f:
                requirements = f.read()

            # Check for key packages
            required_packages = [
                'zenml', 'mlflow', 'optuna', 'evidently', 'prefect',
                'fastapi', 'streamlit', 'pandas', 'scikit-learn'
            ]

            for package in required_packages:
                assert package in requirements.lower()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

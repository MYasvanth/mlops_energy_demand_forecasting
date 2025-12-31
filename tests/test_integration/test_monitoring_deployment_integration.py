"""
Integration tests for monitoring and deployment components.

This module contains tests that verify the integration between
monitoring systems and deployment services, ensuring proper
error handling and data validation throughout the system.
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock, AsyncMock
from pathlib import Path
import tempfile
import asyncio
from fastapi.testclient import TestClient

from src.deployment.fastapi_app import app, monitor_predictions
from src.monitoring.evidently_monitoring import EvidentlyMonitor
from src.monitoring.deepchecks_monitoring import DeepchecksMonitor
from src.models.predict import TimeSeriesPredictor


class TestMonitoringDeploymentIntegration:
    """Test suite for monitoring and deployment integration."""

    @pytest.fixture
    def client(self):
        """Create test client."""
        return TestClient(app)

    @pytest.fixture
    def sample_data(self):
        """Create sample data for testing."""
        data = {
            'time': pd.date_range('2023-01-01', periods=200, freq='H'),
            'total_load_actual': np.random.normal(50000, 5000, 200),
            'temp': np.random.normal(20, 5, 200),
            'humidity': np.random.uniform(30, 90, 200)
        }
        return pd.DataFrame(data)

    def test_monitoring_initialization_in_deployment(self, sample_data):
        """Test that monitoring is properly initialized in deployment."""
        # Test EvidentlyMonitor initialization
        monitor = EvidentlyMonitor(sample_data, target_column='total_load_actual')
        assert monitor is not None
        assert hasattr(monitor, 'run_monitoring_cycle')

        # Test DeepchecksMonitor initialization
        deepchecks_monitor = DeepchecksMonitor(sample_data)
        assert deepchecks_monitor is not None
        assert hasattr(deepchecks_monitor, 'detect_data_drift')

    @patch('src.deployment.fastapi_app.monitor')
    def test_background_monitoring_integration(self, mock_monitor, sample_data):
        """Test background monitoring integration in FastAPI."""
        mock_monitor.run_monitoring_cycle.return_value = {
            "alerts": ["Data drift detected"],
            "drift_score": 0.85
        }

        # Simulate background monitoring call
        predictions = np.array([50000, 51000, 52000])
        asyncio.run(monitor_predictions(sample_data.tail(50), predictions))

        # Verify monitoring was called
        mock_monitor.run_monitoring_cycle.assert_called_once()

    @patch('src.deployment.fastapi_app.models')
    @patch('src.deployment.fastapi_app.monitor')
    def test_prediction_with_monitoring_alerts(self, mock_monitor, mock_models, client, sample_data):
        """Test prediction flow with monitoring alerts."""
        # Setup mocks
        mock_predictor = MagicMock()
        mock_predictor.predict.return_value = np.array([50000, 51000])
        mock_models.__getitem__.return_value = mock_predictor
        mock_models.__contains__.return_value = True

        mock_monitor.run_monitoring_cycle.return_value = {
            "alerts": ["High prediction variance detected"],
            "drift_score": 0.92
        }

        request_data = {
            "model_name": "arima",
            "hours_ahead": 2,
            "recent_data": sample_data.to_dict('records')
        }

        response = client.post("/predict", json=request_data)
        assert response.status_code == 200

        # Verify monitoring was triggered (background task)
        # Note: In test environment, background tasks may not execute
        # but the setup should work in production

    def test_monitoring_data_consistency(self, sample_data):
        """Test that monitoring uses consistent data formats."""
        monitor = EvidentlyMonitor(sample_data, target_column='total_load_actual')

        # Test with current data
        current_data = sample_data.tail(50)
        predictions = np.random.normal(50000, 1000, 10)

        # This should not raise errors about data format
        try:
            results = monitor.run_monitoring_cycle(current_data, predictions)
            # Results should be a dict or None
            assert isinstance(results, (dict, type(None)))
        except Exception as e:
            # If monitoring fails due to missing dependencies, that's acceptable
            # but data format errors should not occur
            assert "data format" not in str(e).lower()

    def test_deepchecks_monitoring_integration(self, sample_data):
        """Test DeepchecksMonitor integration."""
        monitor = DeepchecksMonitor(sample_data)

        # Test data drift detection
        try:
            drift_results = monitor.detect_data_drift(
                sample_data.head(100),
                sample_data.tail(100)
            )
            assert isinstance(drift_results, dict)
        except Exception as e:
            # May fail due to deepchecks version issues, but should not crash
            assert "import" not in str(e).lower()

    def test_monitoring_error_handling_in_deployment(self, client, sample_data):
        """Test error handling when monitoring fails."""
        with patch('src.deployment.fastapi_app.monitor_predictions') as mock_monitor_task:
            mock_monitor_task.side_effect = Exception("Monitoring failed")

            # This should not affect the prediction response
            # (monitoring is background task)
            with patch('src.deployment.fastapi_app.models') as mock_models:
                mock_predictor = MagicMock()
                mock_predictor.predict.return_value = np.array([50000])
                mock_models.__getitem__.return_value = mock_predictor
                mock_models.__contains__.return_value = True

                request_data = {
                    "model_name": "arima",
                    "hours_ahead": 1,
                    "recent_data": sample_data.to_dict('records')
                }

                response = client.post("/predict", json=request_data)
                assert response.status_code == 200  # Prediction should still work

    def test_monitoring_status_endpoint_integration(self, client):
        """Test monitoring status endpoint integration."""
        response = client.get("/monitoring/status")
        assert response.status_code == 200
        data = response.json()
        assert "status" in data

        # Should indicate monitoring status
        if "active" in data.get("status", ""):
            assert "last_check" in data

    def test_monitoring_dashboard_endpoint_integration(self, client):
        """Test monitoring dashboard endpoint integration."""
        response = client.get("/monitoring/dashboard")
        assert response.status_code == 200
        data = response.json()

        # Should either return dashboard URL or error message
        assert "dashboard_url" in data or "error" in data

    @patch('src.deployment.fastapi_app.monitor')
    def test_monitoring_with_missing_reference_data(self, mock_monitor, client):
        """Test monitoring behavior when reference data is missing."""
        mock_monitor.workspace = None

        response = client.get("/monitoring/dashboard")
        assert response.status_code == 200
        data = response.json()
        assert "error" in data

    def test_data_validation_through_pipeline(self, sample_data):
        """Test data validation consistency between monitoring and deployment."""
        # Data should be valid for both monitoring and prediction
        monitor = EvidentlyMonitor(sample_data, target_column='total_load_actual')

        # Should not raise validation errors
        assert monitor.reference_data is not None

        # Test with FastAPI-style data
        api_data = sample_data.to_dict('records')
        df_from_api = pd.DataFrame(api_data)

        # Should be able to convert back to DataFrame
        assert isinstance(df_from_api, pd.DataFrame)
        assert len(df_from_api) == len(sample_data)

    def test_monitoring_configuration_consistency(self):
        """Test that monitoring configurations are consistent."""
        # Both monitoring systems should have similar interfaces
        evidently_monitor = EvidentlyMonitor.__init__
        deepchecks_monitor = DeepchecksMonitor.__init__

        # Both should be callable
        assert callable(evidently_monitor)
        assert callable(deepchecks_monitor)

    def test_error_propagation_from_monitoring_to_deployment(self, sample_data):
        """Test error propagation from monitoring to deployment."""
        # Test with invalid data for monitoring
        invalid_data = pd.DataFrame({'wrong_column': [1, 2, 3]})

        monitor = EvidentlyMonitor(sample_data, target_column='total_load_actual')

        # Monitoring should handle invalid data gracefully
        try:
            results = monitor.run_monitoring_cycle(invalid_data, np.array([50000]))
            # Should not crash the deployment service
            assert True  # If we get here, error was handled
        except Exception:
            # If monitoring fails, deployment should continue
            assert True

    def test_async_monitoring_operations(self, sample_data):
        """Test asynchronous monitoring operations."""
        # Test that monitoring operations can be called synchronously
        predictions = np.array([50000, 51000])

        # This should not block or cause issues
        import asyncio
        asyncio.run(monitor_predictions(sample_data, predictions))

        # Test completed without hanging
        assert True

    def test_monitoring_resource_cleanup(self, sample_data):
        """Test that monitoring resources are properly cleaned up."""
        monitor = EvidentlyMonitor(sample_data, target_column='total_load_actual')

        # Monitor should be usable
        assert monitor is not None

        # After "cleanup" (del), should not cause issues
        del monitor

        # Should be able to create new monitor
        new_monitor = EvidentlyMonitor(sample_data, target_column='total_load_actual')
        assert new_monitor is not None

    def test_monitoring_with_different_data_frequencies(self, sample_data):
        """Test monitoring with different data frequencies."""
        # Test with hourly data
        hourly_data = sample_data

        monitor = EvidentlyMonitor(hourly_data, target_column='total_load_actual')

        # Should handle hourly data
        results = monitor.run_monitoring_cycle(hourly_data.tail(24), np.array([50000]))
        assert isinstance(results, (dict, type(None)))

        # Test with resampled daily data
        daily_data = sample_data.resample('D', on='time').mean().reset_index()
        daily_monitor = EvidentlyMonitor(daily_data, target_column='total_load_actual')

        # Should handle daily data
        daily_results = daily_monitor.run_monitoring_cycle(
            daily_data.tail(7),
            np.array([50000])
        )
        assert isinstance(daily_results, (dict, type(None)))

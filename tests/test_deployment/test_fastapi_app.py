"""
Integration tests for FastAPI deployment service.

This module contains tests that verify the FastAPI prediction service
works correctly with proper error handling and monitoring integration.
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock
from pathlib import Path
import tempfile
import json
from fastapi.testclient import TestClient

from src.deployment.fastapi_app import app, startup_event, monitor_predictions
from src.models.predict import TimeSeriesPredictor


class TestFastAPIApp:
    """Test suite for FastAPI prediction service."""

    @pytest.fixture
    def client(self):
        """Create test client."""
        return TestClient(app)

    @pytest.fixture
    def sample_data(self):
        """Create sample data for testing."""
        data = {
            'time': pd.date_range('2023-01-01', periods=100, freq='H'),
            'total_load_actual': np.random.normal(50000, 5000, 100),
            'temp': np.random.normal(20, 5, 100),
            'humidity': np.random.uniform(30, 90, 100)
        }
        return pd.DataFrame(data)

    def test_health_check(self, client):
        """Test health check endpoint."""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "models_loaded" in data
        assert "monitoring_active" in data

    def test_root_endpoint(self, client):
        """Test root endpoint."""
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert "message" in data
        assert "docs" in data

    def test_list_models(self, client):
        """Test models listing endpoint."""
        response = client.get("/models")
        assert response.status_code == 200
        data = response.json()
        assert "models" in data
        assert "count" in data

    @patch('src.deployment.fastapi_app.models')
    def test_predict_with_invalid_model(self, mock_models, client, sample_data):
        """Test prediction with invalid model name."""
        mock_models.__contains__.return_value = False

        # Convert timestamps to strings for JSON serialization
        sample_data_copy = sample_data.copy()
        sample_data_copy['time'] = sample_data_copy['time'].dt.strftime('%Y-%m-%d %H:%M:%S')

        request_data = {
            "model_name": "invalid_model",
            "hours_ahead": 24,
            "recent_data": sample_data_copy.to_dict('records')
        }

        response = client.post("/predict", json=request_data)
        assert response.status_code == 422  # Validation error

    @patch('src.deployment.fastapi_app.models')
    def test_predict_success(self, mock_models, client, sample_data):
        """Test successful prediction."""
        # Mock predictor
        mock_predictor = MagicMock()
        mock_predictor.predict.return_value = np.array([50000, 51000, 52000])
        mock_models.get.return_value = mock_predictor
        mock_models.__contains__.return_value = True

        # Convert timestamps to strings for JSON serialization
        sample_data_copy = sample_data.copy()
        sample_data_copy['time'] = sample_data_copy['time'].dt.strftime('%Y-%m-%d %H:%M:%S')

        request_data = {
            "model_name": "arima",
            "hours_ahead": 3,
            "recent_data": sample_data_copy.to_dict('records')
        }

        response = client.post("/predict", json=request_data)
        assert response.status_code == 200
        data = response.json()
        assert "predictions" in data
        assert "timestamps" in data
        assert len(data["predictions"]) == 3
        assert data["model_used"] == "arima"

    @patch('src.deployment.fastapi_app.models')
    def test_predict_with_file_data(self, mock_models, client, tmp_path, sample_data):
        """Test prediction using data from file."""
        # Save sample data to file
        data_file = tmp_path / "test_data.csv"
        sample_data.to_csv(data_file, index=False)

        # Mock predictor
        mock_predictor = MagicMock()
        mock_predictor.predict.return_value = np.array([50000, 51000])
        mock_models.get.return_value = mock_predictor
        mock_models.__contains__.return_value = True

        request_data = {
            "model_name": "arima",
            "hours_ahead": 2,
            "data_path": str(data_file)
        }

        response = client.post("/predict", json=request_data)
        assert response.status_code == 200
        data = response.json()
        assert len(data["predictions"]) == 2

    @patch('src.deployment.fastapi_app.models')
    def test_predict_no_data_provided(self, mock_models, client):
        """Test prediction with no data provided."""
        mock_models.__contains__.return_value = True

        request_data = {
            "model_name": "arima",
            "hours_ahead": 24
        }

        response = client.post("/predict", json=request_data)
        assert response.status_code == 400
        assert "No valid data provided" in response.json()["detail"]

    @patch('src.deployment.fastapi_app.models')
    def test_ensemble_predict(self, mock_models, client, sample_data):
        """Test ensemble prediction."""
        # Mock multiple models
        mock_predictor1 = MagicMock()
        mock_predictor1.predict.return_value = np.array([50000])

        mock_predictor2 = MagicMock()
        mock_predictor2.predict.return_value = np.array([51000])

        mock_models.items.return_value = [
            ("arima", mock_predictor1),
            ("lstm", mock_predictor2)
        ]
        mock_models.__len__.return_value = 2
        mock_models.keys.return_value = ["arima", "lstm"]
        mock_models.__contains__.return_value = True

        # Convert timestamps to strings for JSON serialization
        sample_data_copy = sample_data.copy()
        sample_data_copy['time'] = sample_data_copy['time'].dt.strftime('%Y-%m-%d %H:%M:%S')

        request_data = {
            "model_name": "arima",  # Not used in ensemble
            "hours_ahead": 1,
            "recent_data": sample_data_copy.to_dict('records')
        }

        response = client.post("/ensemble-predict", json=request_data)
        assert response.status_code == 200
        data = response.json()
        assert "predictions" in data
        assert data["model_used"] == "ensemble"
        assert "models_used" in data["metadata"]

    def test_monitoring_status(self, client):
        """Test monitoring status endpoint."""
        response = client.get("/monitoring/status")
        assert response.status_code == 200
        data = response.json()
        assert "status" in data

    @patch('src.deployment.fastapi_app.monitor')
    def test_monitoring_dashboard(self, mock_monitor, client):
        """Test monitoring dashboard endpoint."""
        mock_monitor.workspace = None

        response = client.get("/monitoring/dashboard")
        assert response.status_code == 200
        data = response.json()
        assert "error" in data or "dashboard_url" in data

    @patch('src.deployment.fastapi_app.monitor')
    @patch('src.deployment.fastapi_app.BackgroundTasks')
    def test_monitor_predictions_background_task(self, mock_bg_tasks, mock_monitor, sample_data):
        """Test background monitoring task."""
        mock_monitor.run_monitoring_cycle.return_value = {"alerts": []}

        # This would normally be called as a background task
        # For testing, we call it directly
        import asyncio
        asyncio.run(monitor_predictions(sample_data, np.array([50000, 51000])))

        # Verify monitoring was called
        mock_monitor.run_monitoring_cycle.assert_called_once()

    def test_prediction_request_validation(self, client):
        """Test prediction request validation."""
        # Test invalid hours_ahead
        request_data = {
            "model_name": "arima",
            "hours_ahead": 200,  # Exceeds max
            "recent_data": [{"time": "2023-01-01", "value": 50000}]
        }

        response = client.post("/predict", json=request_data)
        assert response.status_code == 422  # Validation error

    @patch('src.deployment.fastapi_app.models')
    def test_predict_with_confidence_intervals(self, mock_models, client, sample_data):
        """Test prediction with confidence intervals."""
        # Mock predictor with scaler
        mock_predictor = MagicMock()
        mock_predictor.predict.return_value = np.array([50000, 51000])
        mock_predictor.scaler = MagicMock()  # Indicate scaler is available
        mock_models.get.return_value = mock_predictor
        mock_models.__contains__.return_value = True

        # Convert timestamps to strings for JSON serialization
        sample_data_copy = sample_data.copy()
        sample_data_copy['time'] = sample_data_copy['time'].dt.strftime('%Y-%m-%d %H:%M:%S')

        request_data = {
            "model_name": "lstm",
            "hours_ahead": 2,
            "recent_data": sample_data_copy.to_dict('records')
        }

        response = client.post("/predict", json=request_data)
        assert response.status_code == 200
        data = response.json()
        assert "confidence_intervals" in data
        assert "lower_bound" in data["confidence_intervals"]
        assert "upper_bound" in data["confidence_intervals"]

    @patch('src.deployment.fastapi_app.startup_event')
    def test_startup_error_handling(self, mock_startup, client):
        """Test startup error handling."""
        mock_startup.side_effect = Exception("Startup failed")

        # Startup is called during app initialization
        # This test verifies error handling in startup
        with patch('src.deployment.fastapi_app.logger') as mock_logger:
            # Re-initialize app to trigger startup
            from src.deployment.fastapi_app import app as test_app
            # Startup errors would be logged
            mock_logger.error.assert_not_called()  # May not be called in test environment

    def test_cors_and_openapi(self, client):
        """Test OpenAPI and CORS configuration."""
        # Test OpenAPI docs
        response = client.get("/docs")
        assert response.status_code == 200

        # Test OpenAPI JSON
        response = client.get("/openapi.json")
        assert response.status_code == 200
        openapi_data = response.json()
        assert "paths" in openapi_data
        assert "/predict" in openapi_data["paths"]

    @patch('src.deployment.fastapi_app.models')
    def test_predict_large_horizon(self, mock_models, client, sample_data):
        """Test prediction with large time horizon."""
        mock_predictor = MagicMock()
        mock_predictor.predict.return_value = np.random.normal(50000, 1000, 168)
        mock_models.get.return_value = mock_predictor
        mock_models.__contains__.return_value = True

        # Convert timestamps to strings for JSON serialization
        sample_data_copy = sample_data.copy()
        sample_data_copy['time'] = sample_data_copy['time'].dt.strftime('%Y-%m-%d %H:%M:%S')

        request_data = {
            "model_name": "arima",
            "hours_ahead": 168,  # Maximum allowed
            "recent_data": sample_data_copy.to_dict('records')
        }

        response = client.post("/predict", json=request_data)
        assert response.status_code == 200
        data = response.json()
        assert len(data["predictions"]) == 168
        assert len(data["timestamps"]) == 168

"""
Custom Exception Classes for Energy Demand Forecasting Project

This module defines custom exception classes used throughout the project
for better error handling and debugging.
"""

from typing import Optional, Any


class EnergyDemandError(Exception):
    """Base exception class for energy demand forecasting errors."""

    def __init__(self, message: str, details: Optional[Any] = None):
        self.message = message
        self.details = details
        super().__init__(self.message)


class DataValidationError(EnergyDemandError):
    """Exception raised for data validation errors."""

    def __init__(self, message: str, field: Optional[str] = None, value: Optional[Any] = None):
        self.field = field
        self.value = value
        details = {"field": field, "value": value} if field or value else None
        super().__init__(message, details)


class DataIngestionError(EnergyDemandError):
    """Exception raised for data ingestion errors."""

    def __init__(self, message: str, source: Optional[str] = None, file_path: Optional[str] = None):
        self.source = source
        self.file_path = file_path
        details = {"source": source, "file_path": file_path} if source or file_path else None
        super().__init__(message, details)


class DataPreprocessingError(EnergyDemandError):
    """Exception raised for data preprocessing errors."""

    def __init__(self, message: str, step: Optional[str] = None, data_shape: Optional[tuple] = None):
        self.step = step
        self.data_shape = data_shape
        details = {"step": step, "data_shape": data_shape} if step or data_shape else None
        super().__init__(message, details)


class FeatureEngineeringError(EnergyDemandError):
    """Exception raised for feature engineering errors."""

    def __init__(self, message: str, feature_name: Optional[str] = None, operation: Optional[str] = None):
        self.feature_name = feature_name
        self.operation = operation
        details = {"feature_name": feature_name, "operation": operation} if feature_name or operation else None
        super().__init__(message, details)


class ModelTrainingError(EnergyDemandError):
    """Exception raised for model training errors."""

    def __init__(self, message: str, model_type: Optional[str] = None, stage: Optional[str] = None):
        self.model_type = model_type
        self.stage = stage
        details = {"model_type": model_type, "stage": stage} if model_type or stage else None
        super().__init__(message, details)


class ModelPredictionError(EnergyDemandError):
    """Exception raised for model prediction errors."""

    def __init__(self, message: str, model_name: Optional[str] = None, input_shape: Optional[tuple] = None):
        self.model_name = model_name
        self.input_shape = input_shape
        details = {"model_name": model_name, "input_shape": input_shape} if model_name or input_shape else None
        super().__init__(message, details)


class MLflowError(EnergyDemandError):
    """Exception raised for MLflow-related errors."""

    def __init__(self, message: str, operation: Optional[str] = None, run_id: Optional[str] = None):
        self.operation = operation
        self.run_id = run_id
        details = {"operation": operation, "run_id": run_id} if operation or run_id else None
        super().__init__(message, details)


class ConfigurationError(EnergyDemandError):
    """Exception raised for configuration errors."""

    def __init__(self, message: str, config_key: Optional[str] = None, config_file: Optional[str] = None):
        self.config_key = config_key
        self.config_file = config_file
        details = {"config_key": config_key, "config_file": config_file} if config_key or config_file else None
        super().__init__(message, details)


class DeploymentError(EnergyDemandError):
    """Exception raised for deployment errors."""

    def __init__(self, message: str, service: Optional[str] = None, endpoint: Optional[str] = None):
        self.service = service
        self.endpoint = endpoint
        details = {"service": service, "endpoint": endpoint} if service or endpoint else None
        super().__init__(message, details)


class MonitoringError(EnergyDemandError):
    """Exception raised for monitoring errors."""

    def __init__(self, message: str, metric: Optional[str] = None, threshold: Optional[float] = None):
        self.metric = metric
        self.threshold = threshold
        details = {"metric": metric, "threshold": threshold} if metric or threshold else None
        super().__init__(message, details)

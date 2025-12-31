"""
Configuration Models for Energy Demand Forecasting

This module defines Pydantic models for configuration validation
across different components of the MLOps pipeline.
"""

from typing import Dict, List, Optional, Union, Any
from pathlib import Path
from pydantic import BaseModel, Field, validator, root_validator
import os


class ModelConfig(BaseModel):
    """Configuration for ML model training and hyperparameters."""

    # General settings
    target_column: str = Field(..., description="Target column for forecasting")
    date_column: str = Field(..., description="Date/time column name")
    forecast_horizon: int = Field(24, ge=1, le=168, description="Hours ahead to forecast")

    # Model hyperparameters for different algorithms
    models: Dict[str, Dict[str, Any]] = Field(..., description="Model-specific hyperparameters")

    # Optuna hyperparameter tuning settings
    optuna: Dict[str, Any] = Field(..., description="Optuna tuning configuration")

    # Cross-validation settings
    cross_validation: Dict[str, Any] = Field(..., description="Cross-validation configuration")

    # Feature selection
    feature_selection: Dict[str, Any] = Field(..., description="Feature selection settings")

    # Model evaluation metrics
    evaluation: Dict[str, Any] = Field(..., description="Evaluation metrics configuration")

    # Model persistence
    persistence: Dict[str, Any] = Field(..., description="Model persistence settings")

    # Prediction settings
    prediction: Dict[str, Any] = Field(..., description="Prediction configuration")

    @validator('models')
    def validate_models(cls, v):
        """Validate that required models are present."""
        required_models = ['arima', 'prophet', 'lstm']
        if not all(model in v for model in required_models):
            raise ValueError(f"Missing required models: {required_models}")
        return v

    @validator('forecast_horizon')
    def validate_forecast_horizon(cls, v):
        """Validate forecast horizon is reasonable."""
        if v > 168:  # 1 week
            raise ValueError("Forecast horizon should not exceed 168 hours (1 week)")
        return v


class MonitoringConfig(BaseModel):
    """Configuration for monitoring and alerting."""

    # Alert thresholds and rules
    alerting: Dict[str, Any] = Field(..., description="Alerting configuration")

    # Alert rules configuration
    alert_rules: List[Dict[str, Any]] = Field(..., description="Alert rules")

    # Business metrics configuration
    business_metrics: Dict[str, Any] = Field(..., description="Business metrics settings")

    # Monitoring dashboard configuration
    dashboard: Dict[str, Any] = Field(..., description="Dashboard configuration")

    # Evidently monitoring configuration
    evidently: Dict[str, Any] = Field(..., description="Evidently monitoring settings")

    # Logging configuration
    logging: Dict[str, Any] = Field(..., description="Logging configuration")

    # Performance monitoring
    performance: Dict[str, Any] = Field(..., description="Performance monitoring settings")

    @validator('alert_rules')
    def validate_alert_rules(cls, v):
        """Validate alert rules structure."""
        required_fields = ['name', 'metric', 'condition', 'threshold', 'severity']
        for rule in v:
            missing = [field for field in required_fields if field not in rule]
            if missing:
                raise ValueError(f"Alert rule missing required fields: {missing}")
        return v


class PrefectConfig(BaseModel):
    """Configuration for Prefect orchestration."""

    # Flow settings
    flow: Dict[str, Any] = Field(..., description="Flow configuration")

    # Task settings
    tasks: Dict[str, Any] = Field(..., description="Task configuration")

    # Scheduling settings
    scheduling: Dict[str, Any] = Field(..., description="Scheduling configuration")

    # Notification settings
    notifications: Dict[str, Any] = Field(..., description="Notification settings")

    # Logging settings
    logging: Dict[str, Any] = Field(..., description="Logging configuration")

    # Storage settings
    storage: Dict[str, Any] = Field(..., description="Storage configuration")

    # Infrastructure settings
    infrastructure: Dict[str, Any] = Field(..., description="Infrastructure configuration")

    # Monitoring and alerting
    monitoring: Dict[str, Any] = Field(..., description="Monitoring configuration")

    # Environment settings
    environment: Dict[str, Any] = Field(..., description="Environment configuration")

    # Security settings
    security: Dict[str, Any] = Field(..., description="Security configuration")

    # Performance settings
    performance: Dict[str, Any] = Field(..., description="Performance configuration")

    # Error handling
    error_handling: Dict[str, Any] = Field(..., description="Error handling configuration")

    # Cleanup settings
    cleanup: Dict[str, Any] = Field(..., description="Cleanup configuration")


class AppConfig(BaseModel):
    """Main application configuration combining all components."""

    # Environment settings
    environment: str = Field("dev", description="Environment (dev/staging/prod)")
    debug: bool = Field(False, description="Debug mode")

    # Paths
    project_root: Path = Field(default_factory=lambda: Path(__file__).parent.parent.parent,
                              description="Project root directory")
    data_dir: Path = Field(default_factory=lambda: Path("data"), description="Data directory")
    models_dir: Path = Field(default_factory=lambda: Path("models"), description="Models directory")
    logs_dir: Path = Field(default_factory=lambda: Path("logs"), description="Logs directory")
    configs_dir: Path = Field(default_factory=lambda: Path("configs"), description="Configs directory")

    # Component configurations
    model: ModelConfig = Field(..., description="Model configuration")
    monitoring: MonitoringConfig = Field(..., description="Monitoring configuration")
    prefect: PrefectConfig = Field(..., description="Prefect configuration")

    @root_validator(pre=True)
    def set_environment_from_env_var(cls, values):
        """Set environment from ENV environment variable if not provided."""
        if 'environment' not in values:
            values['environment'] = os.getenv('ENV', 'dev')
        return values

    @validator('environment')
    def validate_environment(cls, v):
        """Validate environment is one of allowed values."""
        allowed = ['dev', 'staging', 'prod']
        if v not in allowed:
            raise ValueError(f"Environment must be one of: {allowed}")
        return v

    @validator('project_root', 'data_dir', 'models_dir', 'logs_dir', 'configs_dir')
    def resolve_paths(cls, v):
        """Resolve paths relative to project root."""
        if isinstance(v, str):
            v = Path(v)
        if not v.is_absolute():
            # This will be resolved when the config is loaded
            pass
        return v

    class Config:
        """Pydantic configuration."""
        arbitrary_types_allowed = True
        validate_assignment = True


class ConfigValidationError(Exception):
    """Custom exception for configuration validation errors."""
    pass


def validate_config(config_dict: Dict[str, Any]) -> AppConfig:
    """
    Validate configuration dictionary against Pydantic models.

    Args:
        config_dict: Configuration dictionary to validate

    Returns:
        Validated AppConfig instance

    Raises:
        ConfigValidationError: If validation fails
    """
    try:
        return AppConfig(**config_dict)
    except Exception as e:
        raise ConfigValidationError(f"Configuration validation failed: {e}") from e

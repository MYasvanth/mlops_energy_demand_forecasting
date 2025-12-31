"""
Configuration Loader for Energy Demand Forecasting

This module provides configuration loading using Hydra with environment-specific overrides.
"""

import logging
from pathlib import Path
from typing import Dict, Any, Optional
import os

from omegaconf import DictConfig, OmegaConf
from hydra import compose, initialize_config_dir

# Configure logging
logger = logging.getLogger(__name__)


class ConfigLoader:
    """Configuration loader using Hydra for environment-specific overrides."""

    def __init__(self, config_path: str = "configs", config_name: str = "config"):
        """
        Initialize the configuration loader.

        Args:
            config_path: Path to the configuration directory
            config_name: Name of the main configuration file (without .yaml)
        """
        self.config_path = Path(config_path)
        self.config_name = config_name
        self._hydra_initialized = False

    def _initialize_hydra(self, overrides: Optional[list] = None):
        """Initialize Hydra configuration system with robust error handling."""
        if not self._hydra_initialized:
            try:
                # Convert to absolute path for Hydra
                abs_config_path = self.config_path.resolve()

                # Check if config directory exists
                if not abs_config_path.exists():
                    raise FileNotFoundError(f"Configuration directory not found: {abs_config_path}")

                # Check if main config file exists
                main_config_file = abs_config_path / f"{self.config_name}.yaml"
                if not main_config_file.exists():
                    raise FileNotFoundError(f"Main config file not found: {main_config_file}")

                # Initialize Hydra with config directory
                with initialize_config_dir(config_dir=str(abs_config_path), version_base=None):
                    self.cfg = compose(config_name=self.config_name, overrides=overrides or [])
                    self._hydra_initialized = True
                    logger.info(f"Hydra initialized with config: {self.config_name}")
            except Exception as e:
                logger.error(f"Failed to initialize Hydra: {e}")
                raise

    def load_config(self, environment: Optional[str] = None, overrides: Optional[list] = None) -> Dict[str, Any]:
        """
        Load configuration with environment-specific overrides.

        Args:
            environment: Environment to load (dev/staging/prod). If None, uses ENV var or defaults to dev.
            overrides: Additional Hydra overrides as list of strings (e.g., ["model.forecast_horizon=48"])

        Returns:
            Configuration dictionary

        Raises:
            Exception: If configuration loading fails
        """
        # Determine environment
        if environment is None:
            environment = os.getenv('ENV', 'dev')

        logger.info(f"Loading configuration for environment: {environment}")

        # Prepare Hydra overrides
        hydra_overrides = overrides or []
        hydra_overrides.append(f"environment={environment}")

        try:
            # Initialize Hydra if not already done
            self._initialize_hydra(hydra_overrides)

            # Convert OmegaConf to dict
            config_dict = OmegaConf.to_container(self.cfg, resolve=True)

            # Flatten nested config structures for easier access
            self._flatten_config(config_dict)

            # Resolve paths relative to project root
            self._resolve_paths(config_dict)

            logger.info("Configuration loaded successfully")
            return config_dict

        except Exception as e:
            logger.error(f"Failed to load configuration: {e}")
            raise

    def _resolve_paths(self, config_dict: Dict[str, Any]):
        """Resolve relative paths in configuration."""
        try:
            project_root = Path(__file__).parent.parent.parent

            # Ensure project root exists
            if not project_root.exists():
                raise FileNotFoundError(f"Project root not found: {project_root}")

            # Resolve path fields
            config_dict['project_root'] = str(project_root)
            config_dict['data_dir'] = str(project_root / config_dict.get('data_dir', 'data'))
            config_dict['models_dir'] = str(project_root / config_dict.get('models_dir', 'models'))
            config_dict['logs_dir'] = str(project_root / config_dict.get('logs_dir', 'logs'))
            config_dict['configs_dir'] = str(project_root / config_dict.get('configs_dir', 'configs'))

            logger.info(f"Paths resolved successfully. Project root: {project_root}")
        except Exception as e:
            logger.error(f"Failed to resolve paths: {e}")
            raise

    def _flatten_config(self, config_dict: Dict[str, Any]):
        """Flatten nested config structures for easier access."""
        # Flatten model config
        if 'model' in config_dict and 'model' in config_dict['model']:
            model_section = config_dict['model']['model']
            for key, value in model_section.items():
                config_dict['model'][key] = value
            del config_dict['model']['model']

        # Flatten monitoring config
        if 'monitoring' in config_dict and 'monitoring' in config_dict['monitoring']:
            monitoring_section = config_dict['monitoring']['monitoring']
            for key, value in monitoring_section.items():
                config_dict['monitoring'][key] = value
            del config_dict['monitoring']['monitoring']

        # Flatten prefect config
        if 'prefect' in config_dict and 'prefect' in config_dict['prefect']:
            prefect_section = config_dict['prefect']['prefect']
            for key, value in prefect_section.items():
                config_dict['prefect'][key] = value
            del config_dict['prefect']['prefect']

    def reload_config(self) -> Dict[str, Any]:
        """Reload configuration from disk."""
        self._hydra_initialized = False
        return self.load_config()


# Global config loader instance
config_loader = ConfigLoader()


def load_app_config(environment: Optional[str] = None, overrides: Optional[list] = None) -> Dict[str, Any]:
    """
    Convenience function to load application configuration.

    Args:
        environment: Environment to load (dev/staging/prod)
        overrides: Additional configuration overrides

    Returns:
        Configuration dictionary
    """
    return config_loader.load_config(environment, overrides)


def get_config_value(key: str, environment: Optional[str] = None) -> Any:
    """
    Get a specific configuration value by key.

    Args:
        key: Dot-separated key path (e.g., "model.forecast_horizon")
        environment: Environment to load

    Returns:
        Configuration value
    """
    config = load_app_config(environment)

    # Navigate to the value
    keys = key.split('.')
    value = config
    for k in keys:
        if isinstance(value, dict) and k in value:
            value = value[k]
        else:
            raise KeyError(f"Configuration key '{key}' not found")

    return value


# Backward compatibility functions
def load_model_config(environment: Optional[str] = None) -> Dict[str, Any]:
    """Load model configuration (backward compatibility)."""
    config = load_app_config(environment)
    return config.get('model', {})


def load_monitoring_config(environment: Optional[str] = None) -> Dict[str, Any]:
    """Load monitoring configuration (backward compatibility)."""
    config = load_app_config(environment)
    return config.get('monitoring', {})


def load_prefect_config(environment: Optional[str] = None) -> Dict[str, Any]:
    """Load Prefect configuration (backward compatibility)."""
    config = load_app_config(environment)
    return config.get('prefect', {})


if __name__ == "__main__":
    # Example usage
    try:
        config = load_app_config("dev")
        print(f"Environment: {config.get('environment')}")
        print(f"Debug mode: {config.get('debug')}")
        print(f"Model target column: {config.get('model', {}).get('target_column')}")
    except Exception as e:
        print(f"Error loading config: {e}")

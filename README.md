# MLOps Energy Demand Forecasting

## Overview

This project implements an end-to-end MLOps pipeline for energy demand forecasting using ZenML for pipeline orchestration and Prefect for workflow scheduling. It follows senior ML engineering best practices including reproducible environments, modular code, comprehensive testing, data versioning with DVC, experiment tracking with MLflow/ZenML, robust pipelines, CI/CD, monitoring, and security.

The system forecasts energy demand based on historical data, weather features, and other relevant factors using models like ARIMA, Prophet, LSTM, XGBoost, and LightGBM. It includes exploratory data analysis (EDA) for insights, hyperparameter tuning with Optuna, monitoring with Evidently for drift detection, and interactive dashboards with Streamlit.

## Features

- **Data Ingestion**: Load data from CSV, APIs, and databases with validation.
- **Exploratory Data Analysis (EDA)**: Visualize data, statistical summaries, correlation analysis to identify patterns and outliers.
- **Preprocessing**: Clean, normalize, and handle missing values.
- **Feature Engineering**: Create time series features (lags, rolling statistics, seasonality).
- **Model Training**: Train multiple models (ARIMA, Prophet, LSTM, XGBoost, LightGBM) with hyperparameter tuning using Optuna.
- **Pipeline Orchestration**: ZenML pipelines for reproducible ML workflows.
- **Workflow Scheduling**: Prefect flows for automation, retries, and monitoring.
- **Experiment Tracking**: MLflow integration for model versioning and metrics.
- **Data Versioning**: DVC for tracking data changes.
- **Deployment**: Docker containerization and Kubernetes manifests.
- **Monitoring**: Alerts for model drift and performance degradation.
- **Testing**: Comprehensive unit and integration tests.

## Architecture

```
Data Sources (CSV, API, DB) -> Ingestion -> Preprocessing -> Feature Engineering -> Model Training -> Evaluation -> Deployment -> Monitoring
                                      |           |              |              |              |              |              |
                                   ZenML     Prefect       MLflow         DVC         Docker       Kubernetes   
```

## Usage

### Running the Training Pipeline

```bash
python scripts/training/train_script.py --config configs/model/model_config.yaml
```

### Running Prefect Flows

```bash
python prefect_flows/orchestration_flow.py
```

### Running EDA

```bash
python scripts/eda/eda_script.py
```

### Monitoring

```bash
python scripts/monitoring/run_monitoring.py
```

## Best Practices Implemented

- **Reproducible Environments**: Pinned dependencies, virtual environments.
- **Modular Code**: Type hints, docstrings, separation of concerns.
- **Testing**: Unit tests for all modules, integration tests for pipelines.
- **Data Versioning**: DVC for tracking data changes.
- **Experiment Tracking**: MLflow for model metrics and artifacts.
- **Pipeline Robustness**: ZenML for step-based pipelines, Prefect for scheduling.
- **CI/CD**: GitHub Actions for automated testing and deployment.
- **Monitoring**: Alerts for drift, performance dashboards.

## Configuration

Configuration files are located in `configs/`:

- `model/model_config.yaml`: Model hyperparameters.
- `prefect/prefect_config.yaml`: Prefect settings.
- `monitoring/monitoring_config.yaml`: Monitoring settings.

## Testing

Run tests with:

```bash
pytest tests/
```

## Monitoring

- Dashboards: Streamlit for interactive model predictions and monitoring visualization.
- Logs: Centralized in `logs/`
- Drift Detection: Evidently for data drift detection and model performance monitoring.
- Reports: Generated in `reports/monitoring/`


## License

MIT License.

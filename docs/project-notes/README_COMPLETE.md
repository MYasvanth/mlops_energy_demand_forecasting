# MLOps Energy Demand Forecasting - Complete Implementation

This project implements a comprehensive MLOps pipeline for energy demand forecasting, integrating ZenML, MLflow, Optuna, Evidently, and Prefect for production-ready machine learning.

## 🚀 Quick Start

### Prerequisites
- Python 3.9+
- Git
- Virtual environment

### Installation
```bash
# Clone repository
git clone <repository-url>
cd mlops_energy_demand_forecasting

# Create virtual environment
python -m venv venv
venv\Scripts\activate  # Windows
pip install -r requirements.txt

# Initialize ZenML
zenml init

# Start Prefect server
prefect server start

# Start MLflow tracking
mlflow ui
```

### Basic Usage
```bash
# Run training pipeline
python scripts/training/train_script.py

# Start prediction API
python src/deployment/fastapi_app.py

# Launch monitoring dashboard
streamlit run monitoring/dashboards/prediction_dashboard.py

# Run tests
pytest tests/
```

## 🏗️ Architecture Overview

```
Data Sources → ZenML Pipeline → MLflow Tracking → Prefect Orchestration → FastAPI → Monitoring
     ↓              ↓              ↓              ↓              ↓              ↓
   CSV/API       Feature Eng.   Experiment Mgmt  Scheduling   Predictions   Evidently
   Databases     Model Training Hyperparameter  Retry Logic   Real-time    Dashboards
                 Validation    Model Registry  Notifications  Batch Mode   Alerts
```

## 📊 Core Components

### 1. Data Pipeline (ZenML)
- **Ingestion**: CSV, API, database sources with Pydantic validation
- **Preprocessing**: Missing value imputation, outlier removal, normalization
- **Feature Engineering**: Lag features, rolling statistics, seasonal encoding
- **Integration**: MLflow experiment tracking for all steps

### 2. Model Training (Optuna + MLflow)
- **Algorithms**: ARIMA, Prophet, LSTM with hyperparameter optimization
- **Optimization**: Optuna studies for automated parameter tuning
- **Tracking**: MLflow logging of metrics, parameters, and artifacts
- **Registry**: Model versioning and staging

### 3. Orchestration (Prefect)
- **Workflows**: Scheduled training and prediction pipelines
- **Retries**: Automatic failure recovery and notifications
- **Monitoring**: Flow execution tracking and alerting
- **Scheduling**: Cron-based automated runs

### 4. Monitoring (Evidently)
- **Data Drift**: Statistical detection of distribution changes
- **Model Performance**: Regression metrics and degradation alerts
- **Data Quality**: Automated quality checks and reporting
- **Dashboards**: Interactive monitoring interfaces

### 5. Deployment (FastAPI + Streamlit)
- **API Service**: REST endpoints for real-time predictions
- **Batch Processing**: Large-scale prediction jobs
- **Dashboard**: Interactive visualization and control panel
- **Health Checks**: System monitoring and status endpoints

## 🔧 Configuration

### Model Configuration (`configs/model/model_config.yaml`)
```yaml
model:
  target_column: "total_load_actual"
  forecast_horizon: 24

models:
  arima:
    order: [5, 1, 0]
  prophet:
    changepoint_prior_scale: 0.05
  lstm:
    sequence_length: 24
    hidden_units: [64, 32]

optuna:
  n_trials: 50
  timeout: 3600
```

### Prefect Configuration (`configs/prefect/prefect_config.yaml`)
```yaml
orchestration:
  retries: 3
  retry_delay: 60
  notifications:
    enabled: true
    channels: ["email", "slack"]

scheduling:
  training_interval: "daily"
  monitoring_interval: "hourly"
```

## 🚀 Running the Pipeline

### Training Pipeline
```bash
# Direct execution
python scripts/training/train_script.py --config configs/model/model_config.yaml

# ZenML pipeline
python zenml_pipelines/training_pipeline.py

# Prefect orchestrated
python prefect_flows/orchestration_flow.py
```

### Prediction Service
```bash
# Start API server
python src/deployment/fastapi_app.py

# Make prediction request
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "model_name": "lstm",
    "hours_ahead": 24,
    "data_path": "data/processed/recent_data.csv"
  }'
```

### Monitoring Dashboard
```bash
streamlit run monitoring/dashboards/prediction_dashboard.py
```

## 📈 API Endpoints

### Core Endpoints
- `GET /` - API information
- `GET /health` - Health check
- `GET /models` - List available models
- `POST /predict` - Single model predictions
- `POST /ensemble-predict` - Ensemble predictions
- `GET /monitoring/status` - Monitoring status

### Request/Response Examples

**Prediction Request:**
```json
{
  "model_name": "lstm",
  "hours_ahead": 24,
  "data_path": "data/processed/recent_data.csv"
}
```

**Prediction Response:**
```json
{
  "predictions": [1250.5, 1180.3, ...],
  "timestamps": ["2024-01-15T10:00:00", ...],
  "model_used": "lstm",
  "confidence_intervals": {
    "lower_bound": [1150.2, ...],
    "upper_bound": [1350.8, ...]
  },
  "metadata": {
    "prediction_horizon": 24,
    "generated_at": "2024-01-15T09:00:00"
  }
}
```

## 🔍 Monitoring & Alerting

### Data Drift Detection
```python
from src.monitoring.evidently_monitoring import MonitoringPipeline

monitor = MonitoringPipeline("data/processed/reference_data.csv")
monitor.initialize_monitoring()

# Run monitoring cycle
results = monitor.run_monitoring_cycle(current_data, predictions)
print(f"Alerts: {results['alerts']}")
```

### Dashboard Features
- Real-time predictions
- Model performance metrics
- Data drift visualization
- Alert management
- Historical analysis

## 🧪 Testing

### Run All Tests
```bash
# Unit tests
pytest tests/ -v

# Integration tests
pytest tests/test_integration.py -v

# With coverage
pytest --cov=src --cov-report=html
```

### Test Categories
- **Unit Tests**: Individual component testing
- **Integration Tests**: End-to-end pipeline validation
- **Performance Tests**: Model accuracy and speed benchmarks
- **API Tests**: Endpoint functionality and error handling

## 📁 Project Structure

```
mlops_energy_demand_forecasting/
├── configs/                 # Configuration files
│   ├── model/              # Model hyperparameters
│   ├── prefect/            # Orchestration settings
│   └── monitoring/         # Alert configurations
├── data/                   # Data directory
│   ├── raw/               # Raw datasets
│   └── processed/         # Processed data
├── src/                   # Source code
│   ├── data/              # Data processing
│   ├── features/          # Feature engineering
│   ├── models/            # ML models & training
│   ├── monitoring/        # Evidently monitoring
│   └── deployment/        # FastAPI service
├── zenml_pipelines/       # ZenML pipeline definitions
├── prefect_flows/         # Prefect orchestration
├── monitoring/            # Dashboards and alerts
├── scripts/               # Utility scripts
├── tests/                 # Test suite
├── models/                # Trained models
├── reports/               # Generated reports
└── logs/                  # Application logs
```

## 🔐 Security & Best Practices

### Data Security
- Input validation and sanitization
- Secure API authentication (configurable)
- Data encryption at rest and in transit

### Model Security
- Model validation before deployment
- Adversarial input detection
- Regular security audits

### Monitoring Security
- Alert encryption and secure channels
- Access control for monitoring dashboards
- Audit logging for all operations

## 🚀 Deployment Options

### Local Development
```bash
# Start all services
python src/deployment/fastapi_app.py &
streamlit run monitoring/dashboards/prediction_dashboard.py &
prefect server start &
mlflow ui &
```

### Docker Deployment
```bash
# Build and run
docker build -t energy-forecast .
docker run -p 8000:8000 -p 8501:8501 energy-forecast
```

### Cloud Deployment
- **AWS**: ECS/Fargate, Lambda, SageMaker
- **GCP**: Cloud Run, AI Platform, Vertex AI
- **Azure**: Container Instances, Machine Learning

## 📊 Performance Metrics

### Model Performance (Typical Results)
- **MAE**: 150-300 MW (depending on forecast horizon)
- **RMSE**: 200-400 MW
- **R² Score**: 0.75-0.90
- **MAPE**: 8-15%

### System Performance
- **Training Time**: 30-120 minutes (with hyperparameter tuning)
- **Prediction Latency**: <1 second for single predictions
- **API Throughput**: 100+ requests/second
- **Monitoring Overhead**: <5% of total system resources

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make changes with tests
4. Run full test suite
5. Submit pull request

### Development Guidelines
- Follow PEP 8 style guidelines
- Add comprehensive docstrings
- Write tests for new features
- Update documentation
- Ensure CI/CD passes

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 📞 Support

For questions and support:
- Documentation: [Link to docs]
- Issues: [GitHub Issues]
- Discussions: [GitHub Discussions]

## 🔄 Version History

### v1.0.0 - Complete MLOps Implementation
- ✅ ZenML pipeline orchestration
- ✅ MLflow experiment tracking
- ✅ Optuna hyperparameter tuning
- ✅ Evidently monitoring and alerting
- ✅ Prefect workflow orchestration
- ✅ FastAPI prediction service
- ✅ Streamlit monitoring dashboard
- ✅ Comprehensive testing suite
- ✅ Docker containerization
- ✅ Complete documentation

---

**Built with ❤️ using ZenML, MLflow, Optuna, Evidently, and Prefect**

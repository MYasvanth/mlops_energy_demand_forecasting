"""
FastAPI Prediction Service for Energy Demand Forecasting

This module provides a REST API for real-time energy demand predictions
using trained ML models with proper error handling and monitoring.
"""

import logging
from typing import Dict, List, Optional, Union
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
import json
import uvicorn

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, validator
import joblib

# Import project modules
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

try:
    from src.models.predict import TimeSeriesPredictor, EnsemblePredictor
    from src.monitoring.evidently_monitoring import EvidentlyMonitor
    from src.deployment.model_registry import ModelRegistry
except ImportError:
    import os
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
    from src.models.predict import TimeSeriesPredictor, EnsemblePredictor
    from src.monitoring.evidently_monitoring import EvidentlyMonitor
    from src.deployment.model_registry import ModelRegistry

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Energy Demand Forecasting API",
    description="Real-time energy demand prediction service",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Global variables for models and monitoring
models = {}
monitor = None


class PredictionRequest(BaseModel):
    """Request model for predictions."""
    model_name: str = Field(..., description="Model to use for prediction")
    hours_ahead: int = Field(24, ge=1, le=168, description="Hours to forecast ahead")
    recent_data: Optional[List[Dict]] = Field(None, description="Recent data for prediction")
    data_path: Optional[str] = Field(None, description="Path to data file")

    @validator('model_name')
    def validate_model_name(cls, v):
        if v not in models:
            available_models = list(models.keys())
            raise ValueError(f"Model '{v}' not found. Available models: {available_models}")
        return v


class EnsemblePredictionRequest(BaseModel):
    """Request model for ensemble predictions."""
    hours_ahead: int = Field(24, ge=1, le=168, description="Hours to forecast ahead")
    recent_data: Optional[List[Dict]] = Field(None, description="Recent data for prediction")
    data_path: Optional[str] = Field(None, description="Path to data file")


class PredictionResponse(BaseModel):
    """Response model for predictions."""
    predictions: List[float]
    timestamps: List[str]
    model_used: str
    confidence_intervals: Optional[Dict[str, List[float]]] = None
    metadata: Dict


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    timestamp: str
    models_loaded: List[str]
    monitoring_active: bool


@app.on_event("startup")
async def startup_event():
    """Initialize models and monitoring on startup with robust error handling."""
    global models, monitor

    logger.info("Starting Energy Demand Forecasting API...")
    logger.info(f"Loading models from: {ModelRegistry.PRODUCTION_DIR}")

    try:
        # Validate production deployment
        missing = ModelRegistry.validate_deployment()
        if missing:
            logger.warning(f"Missing model files: {missing}")
        
        # Load models using registry
        model_dirs = [ModelRegistry.PRODUCTION_DIR, Path("models")]
        
        loaded_models = 0
        for model_dir in model_dirs:
            if model_dir.exists():
                logger.info(f"Checking for models in {model_dir}")
                
                # Time series models
                ts_models = {
                    'lstm': model_dir / 'lstm_model_lstm.h5',
                    'prophet': model_dir / 'prophet_model_prophet.pkl', 
                    'arima': model_dir / 'arima_model_fitted.pkl'
                }
                
                for model_name, model_file in ts_models.items():
                    if model_file.exists() and model_name not in models:
                        try:
                            predictor = TimeSeriesPredictor(model_name)
                            base_path = str(model_dir / f"{model_name}_model")
                            predictor.load_model(base_path)
                            models[model_name] = predictor
                            loaded_models += 1
                            logger.info(f"Successfully loaded model: {model_name}")
                        except Exception as e:
                            logger.warning(f"Failed to load {model_name}: {e}")
                
                # GBM models
                gbm_models = ['lightgbm', 'xgboost']
                for model_name in gbm_models:
                    model_file = model_dir / f"{model_name}_gbm_model.joblib"
                    if model_file.exists() and model_name not in models:
                        try:
                            import joblib
                            model = joblib.load(model_file)
                            models[model_name] = model
                            loaded_models += 1
                            logger.info(f"Successfully loaded model: {model_name}")
                        except Exception as e:
                            logger.warning(f"Failed to load {model_name}: {e}")

        if loaded_models == 0:
            logger.warning("No models were successfully loaded. API will start with limited functionality.")
        else:
            logger.info(f"Successfully loaded {loaded_models} models: {list(models.keys())}")

        # Initialize monitoring with error handling
        try:
            reference_data_paths = [
                "data/processed/processed_energy_weather.csv",
                "data/processed/current_batch.csv",
                "data/raw/energy_dataset.csv"  # Fallback
            ]

            reference_data = None
            for ref_path in reference_data_paths:
                if Path(ref_path).exists():
                    try:
                        reference_data = pd.read_csv(ref_path, nrows=1000)  # Limit for memory
                        if len(reference_data) > 100:  # Ensure minimum data
                            logger.info(f"Loaded reference data from {ref_path}")
                            break
                    except Exception as e:
                        logger.warning(f"Failed to load reference data from {ref_path}: {e}")
                        continue

            if reference_data is not None and len(reference_data) > 100:
                # Prepare reference data
                if 'time' in reference_data.columns:
                    reference_data['time'] = pd.to_datetime(reference_data['time'], errors='coerce')
                    reference_data = reference_data.dropna(subset=['time'])
                    reference_data.set_index('time', inplace=True)

                # Ensure we have the target column
                if 'total_load_actual' not in reference_data.columns:
                    logger.warning("Target column 'total_load_actual' not found in reference data")
                    reference_data = None
                else:
                    monitor = EvidentlyMonitor(reference_data)
                    logger.info("Monitoring initialized successfully")
            else:
                logger.warning("Insufficient reference data for monitoring initialization")

        except Exception as e:
            logger.error(f"Failed to initialize monitoring: {e}")
            monitor = None

        logger.info(f"API startup complete. Loaded {len(models)} models: {list(models.keys())}. Monitoring active: {monitor is not None}")

    except Exception as e:
        logger.error(f"Critical error during startup: {e}")
        # Don't fail startup, but log the error
        logger.warning("API starting with degraded functionality due to startup errors")


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy",
        timestamp=datetime.now().isoformat(),
        models_loaded=list(models.keys()),
        monitoring_active=monitor is not None
    )


@app.post("/predict", response_model=PredictionResponse)
async def predict_demand(request: PredictionRequest, background_tasks: BackgroundTasks):
    """
    Generate energy demand predictions.

    This endpoint accepts prediction requests and returns forecasted values
    with optional confidence intervals and metadata.
    """
    try:
        logger.info(f"Prediction request: model={request.model_name}, hours={request.hours_ahead}")

        # Get predictor
        predictor = models.get(request.model_name)
        if not predictor:
            raise HTTPException(status_code=404, detail=f"Model '{request.model_name}' not found")

        # Prepare data for prediction
        if request.recent_data:
            # Use provided data
            data_df = pd.DataFrame(request.recent_data)
            if 'time' in data_df.columns:
                data_df['time'] = pd.to_datetime(data_df['time'])
                data_df.set_index('time', inplace=True)
        elif request.data_path and Path(request.data_path).exists():
            # Load from file
            data_df = pd.read_csv(request.data_path)
            if 'time' in data_df.columns:
                data_df['time'] = pd.to_datetime(data_df['time'])
                data_df.set_index('time', inplace=True)
        else:
            raise HTTPException(status_code=400, detail="No valid data provided")

        # Generate predictions
        predictions = predictor.predict(data_df, steps=request.hours_ahead)

        # Create timestamps for predictions
        if isinstance(data_df.index, pd.DatetimeIndex):
            last_time = data_df.index[-1]
            timestamps = [
                (last_time + timedelta(hours=i+1)).isoformat()
                for i in range(request.hours_ahead)
            ]
        else:
            # Fallback if index is not datetime
            base_time = datetime.now()
            timestamps = [
                (base_time + timedelta(hours=i+1)).isoformat()
                for i in range(request.hours_ahead)
            ]

        # Calculate confidence intervals (simplified)
        confidence_intervals = None
        if hasattr(predictor, 'scaler'):
            # For LSTM models with scalers
            pred_std = np.std(predictions) * 0.1  # Simplified confidence calculation
            confidence_intervals = {
                "lower_bound": (predictions - 1.96 * pred_std).tolist(),
                "upper_bound": (predictions + 1.96 * pred_std).tolist()
            }

        # Background monitoring
        if monitor is not None:
            background_tasks.add_task(monitor_predictions, data_df, predictions)

        response = PredictionResponse(
            predictions=predictions.tolist(),
            timestamps=timestamps,
            model_used=request.model_name,
            confidence_intervals=confidence_intervals,
            metadata={
                "prediction_horizon": request.hours_ahead,
                "data_points_used": len(data_df),
                "generated_at": datetime.now().isoformat()
            }
        )

        logger.info(f"Prediction completed: {len(predictions)} values generated")
        return response

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/models")
async def list_models():
    """List available models with metadata."""
    production_models = ModelRegistry.get_production_models()
    metadata = ModelRegistry.get_metadata()
    
    return {
        "models": list(models.keys()),
        "count": len(models),
        "production_models": production_models,
        "deployment_metadata": metadata
    }


@app.get("/models/registry")
async def get_registry_info():
    """Get complete model registry information."""
    return {
        "registry": ModelRegistry.MODEL_CONFIGS,
        "production_models": ModelRegistry.get_production_models(),
        "metadata": ModelRegistry.get_metadata(),
        "validation": {
            "missing_files": ModelRegistry.validate_deployment(),
            "status": "valid" if not ModelRegistry.validate_deployment() else "incomplete"
        }
    }


@app.post("/ensemble-predict")
async def ensemble_predict(request: EnsemblePredictionRequest):
    """Generate ensemble predictions using multiple models."""
    try:
        # Create ensemble with available models
        model_configs = []
        for model_name, predictor in models.items():
            model_configs.append({
                'model_type': model_name,
                'model_path': f"models/production/{model_name}_model",
                'weight': 1.0 / len(models) if len(models) > 0 else 1.0  # Equal weights, avoid division by zero
            })

        if not model_configs:
            raise HTTPException(status_code=404, detail="No models available for ensemble prediction")

        ensemble = EnsemblePredictor(model_configs)

        # Prepare data (same as single prediction)
        if request.recent_data:
            data_df = pd.DataFrame(request.recent_data)
            if 'time' in data_df.columns:
                data_df['time'] = pd.to_datetime(data_df['time'])
                data_df.set_index('time', inplace=True)
        elif request.data_path and Path(request.data_path).exists():
            data_df = pd.read_csv(request.data_path)
            if 'time' in data_df.columns:
                data_df['time'] = pd.to_datetime(data_df['time'])
                data_df.set_index('time', inplace=True)
        else:
            raise HTTPException(status_code=400, detail="No valid data provided")

        # Generate ensemble predictions
        predictions = ensemble.predict(data_df, steps=request.hours_ahead)

        if isinstance(data_df.index, pd.DatetimeIndex):
            last_time = data_df.index[-1]
            timestamps = [
                (last_time + timedelta(hours=i+1)).isoformat()
                for i in range(request.hours_ahead)
            ]
        else:
            base_time = datetime.now()
            timestamps = [
                (base_time + timedelta(hours=i+1)).isoformat()
                for i in range(request.hours_ahead)
            ]

        return PredictionResponse(
            predictions=predictions.tolist(),
            timestamps=timestamps,
            model_used="ensemble",
            metadata={
                "models_used": list(models.keys()),
                "prediction_horizon": request.hours_ahead,
                "generated_at": datetime.now().isoformat()
            }
        )

    except Exception as e:
        logger.error(f"Ensemble prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/monitoring/status")
async def monitoring_status():
    """Get monitoring status and recent alerts."""
    if monitor is None:
        return {"status": "not_initialized"}

    # Get recent monitoring results (simplified)
    return {
        "status": "active",
        "last_check": datetime.now().isoformat(),
        "alerts": []  # Would be populated from monitoring system
    }


@app.get("/monitoring/dashboard")
async def monitoring_dashboard():
    """Get Evidently monitoring dashboard URL."""
    try:
        if monitor is None:
            return {"error": "Monitoring not initialized"}

        if monitor.workspace is None:
            return {"error": "Evidently workspace not available. Using simplified monitoring."}

        # Get dashboard URL from workspace
        projects = monitor.workspace.list_projects()
        if not projects:
            return {"error": "No projects found in workspace"}

        dashboard_url = monitor.workspace.get_dashboard(projects[0].id)
        return {
            "dashboard_url": dashboard_url,
            "workspace": "energy_demand_monitoring",
            "projects": [p.name for p in projects]
        }
    except Exception as e:
        logger.error(f"Dashboard access error: {e}")
        return {"error": f"Dashboard not available: {str(e)}"}


async def monitor_predictions(data_df: pd.DataFrame, predictions: np.ndarray):
    """Background task to monitor predictions."""
    try:
        if monitor is not None:
            # Run monitoring cycle
            results = monitor.run_monitoring_cycle(
                data_df.tail(100),  # Use recent data
                predictions[:24],   # Monitor first 24 hours
                output_dir="reports/monitoring/api"
            )

            # Log alerts
            if results.get('alerts'):
                logger.warning(f"Monitoring alerts: {results['alerts']}")

    except Exception as e:
        logger.error(f"Monitoring error: {e}")


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "message": "Energy Demand Forecasting API",
        "docs": "/docs",
        "health": "/health"
    }


if __name__ == "__main__":
    uvicorn.run(
        "src.deployment.fastapi_app:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )

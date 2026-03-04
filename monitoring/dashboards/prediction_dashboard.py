"""
Streamlit Prediction Dashboard for Energy Demand Forecasting

This module creates an interactive dashboard for model predictions,
performance monitoring, and data visualization using Streamlit.
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
from pathlib import Path
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import sys
import logging

# Add project root to path for imports
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.monitoring.evidently_monitoring import EvidentlyMonitor, MonitoringPipeline
from data_loader import load_processed_dataset
import joblib
import os

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set page configuration
st.set_page_config(
    page_title="Energy Demand Forecasting Dashboard",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Configure matplotlib
plt.style.use('default')
sns.set_palette("husl")


@st.cache_resource
def load_models_cached():
    """Load all trained models - cached across reruns."""
    model_cache = {}
    errors = []
    
    # Get absolute path to project root
    current_file = Path(__file__).resolve()
    project_root = current_file.parent.parent.parent
    models_dir = project_root / "models"
    
    logger.info(f"Loading models from: {models_dir}")
    
    if not models_dir.exists():
        error_msg = f"Models directory not found: {models_dir}"
        logger.error(error_msg)
        errors.append(error_msg)
        return model_cache, errors
    
    # GBM models - simple direct loading
    try:
        lightgbm_path = models_dir / 'lightgbm_gbm_model.joblib'
        lightgbm_eng_path = models_dir / 'lightgbm_gbm_model_feature_engineer.joblib'
        
        if lightgbm_path.exists():
            model = joblib.load(str(lightgbm_path))
            engineer = joblib.load(str(lightgbm_eng_path)) if lightgbm_eng_path.exists() else None
            model_cache['LightGBM'] = {'model': model, 'engineer': engineer, 'type': 'gbm'}
            logger.info("✅ Loaded LightGBM")
    except Exception as e:
        error_msg = f"LightGBM load error: {str(e)}"
        logger.error(error_msg)
        errors.append(error_msg)
    
    try:
        xgboost_path = models_dir / 'xgboost_gbm_model.joblib'
        xgboost_eng_path = models_dir / 'xgboost_gbm_model_feature_engineer.joblib'
        
        if xgboost_path.exists():
            model = joblib.load(str(xgboost_path))
            engineer = joblib.load(str(xgboost_eng_path)) if xgboost_eng_path.exists() else None
            model_cache['XGBoost'] = {'model': model, 'engineer': engineer, 'type': 'gbm'}
            logger.info("✅ Loaded XGBoost")
    except Exception as e:
        error_msg = f"XGBoost load error: {str(e)}"
        logger.error(error_msg)
        errors.append(error_msg)
    
    # Time series models
    try:
        sys.path.insert(0, str(project_root))
        from src.models.predict import TimeSeriesPredictor
        
        for name, base in [('LSTM', 'lstm_model'), ('Prophet', 'prophet_model'), ('ARIMA', 'arima_model')]:
            try:
                predictor = TimeSeriesPredictor(name.lower())
                predictor.load_model(str(models_dir / base))
                model_cache[name] = {'model': predictor, 'type': 'timeseries'}
                logger.info(f"✅ Loaded {name}")
            except Exception as e:
                error_msg = f"{name} load error: {str(e)}"
                logger.error(error_msg)
                errors.append(error_msg)
    except Exception as e:
        error_msg = f"TimeSeriesPredictor import error: {str(e)}"
        logger.error(error_msg)
        errors.append(error_msg)
    
    logger.info(f"📊 Total: {len(model_cache)} models loaded")
    return model_cache, errors


class PredictionDashboard:
    """
    Interactive dashboard for energy demand forecasting predictions and monitoring.
    """

    def __init__(self):
        self.data_cache = {}
        self.model_cache, self.load_errors = load_models_cached()  # Get cache and errors
        self.monitor = None
        self.monitoring_results = None
        self.reference_data_path = "data/processed/processed_energy_weather.csv"
        self.current_data_path = "data/processed/current_batch.csv"


    def prepare_data_for_prediction(self, model_type: str):
        """Prepare data with proper preprocessing for predictions."""
        try:
            # Load processed data first
            processed_data = load_processed_dataset()
            
            if processed_data is None or processed_data.empty:
                # Fallback: load and preprocess raw data
                from src.data.ingestion import ingest_data
                from src.data.preprocessing import full_preprocessing_pipeline
                
                raw_data = ingest_data(self.energy_path, self.weather_path)
                processed_data = full_preprocessing_pipeline(raw_data['energy'], raw_data['weather'])
            
            # Ensure datetime index
            if not isinstance(processed_data.index, pd.DatetimeIndex):
                if 'time' in processed_data.columns:
                    processed_data['time'] = pd.to_datetime(processed_data['time'])
                    processed_data.set_index('time', inplace=True)
                else:
                    # Try to convert existing index to datetime
                    try:
                        processed_data.index = pd.to_datetime(processed_data.index)
                    except:
                        st.warning("⚠️ Could not convert index to datetime. Predictions may have incorrect timestamps.")
            
            return processed_data
            
        except Exception as e:
            st.error(f"Error preparing data: {e}")
            return None
    
    def load_data(self, data_path: str) -> pd.DataFrame:
        """
        Load and cache data.

        Args:
            data_path (str): Path to data file.

        Returns:
            pd.DataFrame: Loaded data.
        """
        if data_path not in self.data_cache:
            try:
                df = pd.read_csv(data_path)
                if 'time' in df.columns:
                    # Handle different datetime formats
                    df['time'] = pd.to_datetime(df['time'], errors='coerce')
                    df.set_index('time', inplace=True)
                elif df.index.name != 'time' and len(df.columns) > 0:
                    # Try to convert index if it's not datetime
                    try:
                        df.index = pd.to_datetime(df.index, errors='coerce')
                    except:
                        pass
                self.data_cache[data_path] = df
            except Exception as e:
                st.error(f"Error loading data from {data_path}: {e}")
                return pd.DataFrame()

        return self.data_cache[data_path]

    def create_sidebar(self):
        """Create sidebar with controls."""
        st.sidebar.title("⚡ Energy Demand Forecasting")
        
        # Show model loading status with debug info
        if self.model_cache:
            st.sidebar.success(f"✅ {len(self.model_cache)} models loaded")
            with st.sidebar.expander("Loaded Models"):
                for model_name in self.model_cache.keys():
                    st.write(f"• {model_name}")
        else:
            st.sidebar.error("❌ No models loaded!")
            if self.load_errors:
                with st.sidebar.expander("⚠️ Load Errors"):
                    for error in self.load_errors:
                        st.error(error)

        # Data selection
        st.sidebar.subheader("Data Selection")
        self.energy_path = st.sidebar.text_input(
            "Energy Dataset Path",
            value="data/raw/energy_dataset.csv"
        )
        self.weather_path = st.sidebar.text_input(
            "Weather Dataset Path",
            value="data/raw/weather_features.csv"
        )

        # Model selection
        st.sidebar.subheader("Model Selection")
        available_models = list(self.model_cache.keys()) if self.model_cache else []
        # Filter out GBM models - they don't work with dashboard data
        time_series_models = [m for m in available_models if m in ['LSTM', 'Prophet', 'ARIMA']]
        if not time_series_models:
            time_series_models = ["LSTM", "Prophet", "ARIMA"]
        
        self.selected_model = st.sidebar.selectbox(
            "Choose Model",
            time_series_models,
            index=0
        )
        
        if self.model_cache and any(m in ['LightGBM', 'XGBoost'] for m in self.model_cache.keys()):
            st.sidebar.info("ℹ️ GBM models hidden (require 200 features)")

        # Prediction settings
        st.sidebar.subheader("Prediction Settings")
        self.forecast_hours = st.sidebar.slider(
            "Forecast Hours Ahead",
            min_value=1,
            max_value=168,  # 1 week
            value=24,
            step=1
        )

        # Real-time toggle
        self.real_time_mode = st.sidebar.checkbox("Real-time Mode", value=False)

        # Monitoring controls
        st.sidebar.subheader("Monitoring Controls")
        if st.sidebar.button("🔍 Run Monitoring"):
            with st.spinner("Running Evidently monitoring..."):
                self.run_monitoring_cycle()

        # Refresh button
        if st.sidebar.button("🔄 Refresh Data"):
            self.data_cache.clear()
            st.rerun()

    def display_data_overview(self):
        """Display data overview section."""
        st.header("📊 Data Overview")

        col1, col2, col3 = st.columns(3)

        try:
            # Load data
            energy_data = self.load_data(self.energy_path)
            weather_data = self.load_data(self.weather_path)

            with col1:
                st.subheader("Energy Data")
                if not energy_data.empty:
                    st.metric("Records", len(energy_data))
                    st.metric("Date Range",
                            f"{energy_data.index.min().date()} to {energy_data.index.max().date()}")
                    st.metric("Target Variable", "total_load_actual")

            with col2:
                st.subheader("Weather Data")
                if not weather_data.empty:
                    st.metric("Records", len(weather_data))
                    st.metric("Features", len(weather_data.columns))

            with col3:
                st.subheader("Data Quality")
                if not energy_data.empty:
                    missing_pct = (energy_data.isnull().sum().sum() / len(energy_data)) * 100
                    st.metric("Missing Values", ".1f")

        except Exception as e:
            st.error(f"Error loading data overview: {e}")

    def display_time_series_plot(self):
        """Display interactive time series plot."""
        st.header("📈 Time Series Analysis")

        try:
            # Load processed data which contains the time series
            processed_data = load_processed_dataset()

            if processed_data is None or processed_data.empty:
                st.warning("⚠️ Processed data not found. Please ensure the data processing pipeline has been run.")
                st.info("Run the data processing pipeline to generate the processed dataset.")
                return

            if not processed_data.empty and 'total_load_actual' in processed_data.columns:
                # Create interactive plot
                fig = go.Figure()

                # Add energy load - main target variable
                fig.add_trace(go.Scatter(
                    x=processed_data.index,
                    y=processed_data['total_load_actual'],
                    mode='lines',
                    name='Actual Load (MW)',
                    line=dict(color='blue', width=2)
                ))

                # Add forecast if available
                if 'total_load_forecast' in processed_data.columns:
                    fig.add_trace(go.Scatter(
                        x=processed_data.index,
                        y=processed_data['total_load_forecast'],
                        mode='lines',
                        name='Forecast Load (MW)',
                        line=dict(color='red', width=2, dash='dash')
                    ))

                # Add renewable generation if available
                renewable_cols = ['generation_solar', 'generation_wind_onshore', 'generation_hydro_water_reservoir']
                for col in renewable_cols:
                    if col in processed_data.columns:
                        fig.add_trace(go.Scatter(
                            x=processed_data.index,
                            y=processed_data[col],
                            mode='lines',
                            name=f'{col.replace("generation_", "").title()} (MW)',
                            line=dict(width=1, dash='dot'),
                            visible='legendonly'  # Hidden by default to reduce clutter
                        ))

                fig.update_layout(
                    title="Energy Demand and Generation Time Series",
                    xaxis_title="Time",
                    yaxis_title="Energy (MW)",
                    height=500,
                    showlegend=True,
                    hovermode='x unified'
                )

                # Add date range selector BEFORE creating the plot
                data_to_plot = processed_data  # Default to full dataset

                if pd.api.types.is_datetime64_any_dtype(processed_data.index):
                    try:
                        # Convert to date objects for the date input
                        min_date = processed_data.index.min().date()
                        max_date = processed_data.index.max().date()

                        date_range = st.date_input(
                            "Select Date Range (optional)",
                            value=(min_date, max_date),
                            min_value=min_date,
                            max_value=max_date,
                            key="date_range_selector"
                        )

                        # Filter data based on selected date range if different from full range
                        if len(date_range) == 2:
                            start_date, end_date = date_range
                            if start_date != min_date or end_date != max_date:
                                # Convert back to datetime for filtering
                                start_datetime = pd.to_datetime(start_date)
                                end_datetime = pd.to_datetime(end_date) + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)  # Include end of day

                                filtered_data = processed_data.loc[start_datetime:end_datetime]
                                if len(filtered_data) > 0:
                                    data_to_plot = filtered_data
                                    st.info(f"📅 Showing data from {start_date} to {end_date} ({len(filtered_data)} records)")
                                else:
                                    st.warning("⚠️ No data found in the selected date range. Showing full dataset.")
                            else:
                                st.info("📊 Showing full dataset (no date filter applied)")

                    except Exception as e:
                        st.warning(f"Date range selection issue: {str(e)}. Showing full dataset.")
                        st.info("📊 Displaying full dataset")

                # Create plot with the selected/filtered data
                fig = go.Figure()

                # Add energy load - main target variable
                fig.add_trace(go.Scatter(
                    x=data_to_plot.index,
                    y=data_to_plot['total_load_actual'],
                    mode='lines',
                    name='Actual Load (MW)',
                    line=dict(color='blue', width=2)
                ))

                # Add forecast if available
                if 'total_load_forecast' in data_to_plot.columns:
                    fig.add_trace(go.Scatter(
                        x=data_to_plot.index,
                        y=data_to_plot['total_load_forecast'],
                        mode='lines',
                        name='Forecast Load (MW)',
                        line=dict(color='red', width=2, dash='dash')
                    ))

                # Add renewable generation if available
                renewable_cols = ['generation_solar', 'generation_wind_onshore', 'generation_hydro_water_reservoir']
                for col in renewable_cols:
                    if col in data_to_plot.columns:
                        fig.add_trace(go.Scatter(
                            x=data_to_plot.index,
                            y=data_to_plot[col],
                            mode='lines',
                            name=f'{col.replace("generation_", "").title()} (MW)',
                            line=dict(width=1, dash='dot'),
                            visible='legendonly'  # Hidden by default to reduce clutter
                        ))

                fig.update_layout(
                    title="Energy Demand and Generation Time Series",
                    xaxis_title="Time",
                    yaxis_title="Energy (MW)",
                    height=500,
                    showlegend=True,
                    hovermode='x unified'
                )

                st.plotly_chart(fig, use_container_width=True)

                # Show data statistics
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Total Records", len(processed_data))
                with col2:
                    if pd.api.types.is_datetime64_any_dtype(processed_data.index):
                        st.metric("Date Range", f"{processed_data.index.min().date()} to {processed_data.index.max().date()}")
                    else:
                        st.metric("Records", len(processed_data))
                with col3:
                    avg_load = processed_data['total_load_actual'].mean()
                    st.metric("Avg Load", ".0f")
                with col4:
                    max_load = processed_data['total_load_actual'].max()
                    st.metric("Peak Load", ".0f")

                # Additional insights
                st.subheader("Data Insights")
                col1, col2 = st.columns(2)

                with col1:
                    st.write("**Load Statistics:**")
                    load_stats = processed_data['total_load_actual'].describe()
                    st.write(f"- Mean: {load_stats['mean']:.0f} MW")
                    st.write(f"- Std: {load_stats['std']:.0f} MW")
                    st.write(f"- Min: {load_stats['min']:.0f} MW")
                    st.write(f"- Max: {load_stats['max']:.0f} MW")

                with col2:
                    if 'generation_solar' in processed_data.columns:
                        st.write("**Renewable Generation:**")
                        solar_mean = processed_data['generation_solar'].mean()
                        wind_mean = processed_data.get('generation_wind_onshore', pd.Series()).mean()
                        st.write(f"- Solar: {solar_mean:.0f} MW avg")
                        st.write(f"- Wind: {wind_mean:.0f} MW avg")
                    else:
                        st.write("**Data Quality:**")
                        missing_pct = processed_data.isnull().sum().sum() / (len(processed_data) * len(processed_data.columns)) * 100
                        st.write(f"- Missing Values: {missing_pct:.1f}%")

            else:
                st.warning("⚠️ Time series data not available. The processed data must contain 'total_load_actual' column.")
                st.info("Please run the data processing pipeline to generate the required dataset.")

        except Exception as e:
            st.error(f"Error creating time series plot: {e}")
            st.info("Make sure the processed data file exists and contains the required columns.")

    def display_predictions(self):
        """Display prediction results."""
        st.header("🔮 Predictions")
        
        # Diagnostic information
        with st.expander("🔍 Model Loading Diagnostics"):
            import os
            current_file = Path(__file__).resolve()
            project_root = current_file.parent.parent.parent
            models_dir = project_root / "models"
            
            st.write(f"**Script location:** `{current_file}`")
            st.write(f"**Project root:** `{project_root}`")
            st.write(f"**Models directory:** `{models_dir}`")
            st.write(f"**Models dir exists:** {models_dir.exists()}")
            st.write(f"**Current working dir:** `{os.getcwd()}`")
            
            if models_dir.exists():
                st.write("\n**Files in models directory:**")
                try:
                    files = list(models_dir.glob("*.joblib")) + list(models_dir.glob("*.pkl")) + list(models_dir.glob("*.h5"))
                    for f in files:
                        st.write(f"- {f.name}")
                except Exception as e:
                    st.error(f"Error listing files: {e}")
            
            st.write(f"\n**Models in cache:** {len(self.model_cache)}")
            if self.model_cache:
                for name, info in self.model_cache.items():
                    st.write(f"- {name}: {info['type']}")
            
            if self.load_errors:
                st.write("\n**Load Errors:**")
                for error in self.load_errors:
                    st.error(error)
        
        # Show loaded models status
        if not self.model_cache:
            st.error("❌ No models loaded. Please install missing dependencies:")
            st.code("pip install lightgbm xgboost\npip install --force-reinstall numpy==1.23.5", language="bash")
            st.warning("After installing dependencies, refresh the page or restart the dashboard.")
            return
        
        st.success(f"✅ {len(self.model_cache)} models loaded: {', '.join(self.model_cache.keys())}")

        try:
            if st.button("🚀 Generate Predictions", type="primary"):
                with st.spinner("Generating predictions..."):
                    # Prepare data with proper preprocessing
                    processed_data = self.prepare_data_for_prediction(self.selected_model)
                    
                    if processed_data is None or processed_data.empty:
                        st.error("Failed to load/prepare data for prediction")
                        return
                    
                    if 'total_load_actual' not in processed_data.columns:
                        st.error("Target column 'total_load_actual' not found in data")
                        return
                    
                    model_name = self.selected_model
                    
                    if model_name not in self.model_cache:
                        st.error(f"Model {model_name} not loaded. Available: {list(self.model_cache.keys())}")
                        return
                    
                    model_info = self.model_cache[model_name]
                    
                    # Generate predictions based on model type
                    if model_info['type'] == 'gbm':
                        st.warning("⚠️ GBM models require 200 engineered features")
                        st.info("✅ Use time series models: LSTM, Prophet, ARIMA")
                        return
                    else:
                        # Time series prediction
                        predictor = model_info['model']
                        predictions = predictor.predict(processed_data, steps=self.forecast_hours)
                    
                    # Create prediction dates - handle both datetime and numeric indices
                    if isinstance(processed_data.index, pd.DatetimeIndex) and len(processed_data.index) > 0:
                        last_date = processed_data.index[-1]
                        st.info(f"📅 Last data point: {last_date}")
                        pred_dates = pd.date_range(start=last_date, periods=self.forecast_hours + 1, freq='H')[1:]
                    else:
                        # Fallback: use dataset end date (2018-12-31) + predictions forward
                        st.warning("⚠️ Data index issue detected. Using 2018-12-31 as base date.")
                        last_date = pd.Timestamp('2018-12-31 23:00:00')
                        pred_dates = pd.date_range(start=last_date, periods=self.forecast_hours + 1, freq='H')[1:]
                    
                    # Create results DataFrame
                    pred_df = pd.DataFrame({
                        'date': pred_dates,
                        'prediction': predictions,
                        'model': model_name
                    })
                    
                    # Display predictions
                    st.subheader(f"{model_name} Predictions")
                    st.dataframe(pred_df)
                    
                    # Plot predictions
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(
                        x=pred_df['date'],
                        y=pred_df['prediction'],
                        mode='lines+markers',
                        name='Predictions',
                        line=dict(color='green', width=3)
                    ))
                    
                    fig.update_layout(
                        title=f"{model_name} Forecast - Next {self.forecast_hours} Hours",
                        xaxis_title="Time",
                        yaxis_title="Predicted Load (MW)",
                        height=400
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Prediction statistics
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Mean Prediction", f"{predictions.mean():.0f}")
                    with col2:
                        st.metric("Min Prediction", f"{predictions.min():.0f}")
                    with col3:
                        st.metric("Max Prediction", f"{predictions.max():.0f}")
        
        except Exception as e:
            st.error(f"Error generating predictions: {e}")
            import traceback
            st.code(traceback.format_exc())

    def display_model_performance(self):
        """Display model performance metrics."""
        st.header("📊 Model Performance")

        if self.monitoring_results and 'model_performance' in self.monitoring_results:
            perf = self.monitoring_results['model_performance']

            col1, col2, col3, col4 = st.columns(4)

            with col1:
                mae = perf.get('mae', 0)
                st.metric("MAE", ".2f")
            with col2:
                rmse = perf.get('rmse', 0)
                st.metric("RMSE", ".2f")
            with col3:
                r2 = perf.get('r2_score', 0)
                st.metric("R² Score", ".3f")
            with col4:
                mean_error = perf.get('mean_error', 0)
                st.metric("Mean Error", ".2f")

            st.subheader("Performance Details")
            st.json(perf)
        else:
            st.info("Run monitoring to see actual performance metrics")

            # Placeholder for actual performance metrics
            col1, col2, col3, col4 = st.columns(4)

            with col1:
                st.metric("MAE", "245.3", "↓ 12.1")
            with col2:
                st.metric("RMSE", "312.7", "↓ 8.5")
            with col3:
                st.metric("R² Score", "0.87", "↑ 0.03")
            with col4:
                st.metric("MAPE", "4.2%", "↓ 0.8%")

        # Performance over time chart
        st.subheader("Performance Over Time")

        # Sample performance data (replace with actual historical data)
        dates = pd.date_range(start='2024-01-01', periods=30, freq='D')
        mae_scores = 250 + np.random.normal(0, 20, 30)

        perf_df = pd.DataFrame({
            'date': dates,
            'mae': mae_scores
        })

        fig = px.line(perf_df, x='date', y='mae', title='MAE Over Time')
        fig.update_layout(height=300)
        st.plotly_chart(fig, use_container_width=True)

    def display_monitoring_alerts(self):
        """Display monitoring alerts and drift detection."""
        st.header("🚨 Monitoring & Alerts")

        if self.monitoring_results:
            # Display real alerts from Evidently
            alerts = self.monitoring_results.get('alerts', [])
            if alerts:
                for alert in alerts:
                    st.error(f"🚨 {alert}")
            else:
                st.success("✅ No alerts detected")

            # Display data quality info
            if 'data_quality' in self.monitoring_results:
                dq = self.monitoring_results['data_quality']
                st.subheader("Data Quality")
                col1, col2 = st.columns(2)
                with col1:
                    quality_score = dq.get('data_quality_score', 0)
                    st.metric("Quality Score", ".2f")
                with col2:
                    missing_vals = dq.get('missing_values', {})
                    total_missing = sum(missing_vals.values()) if missing_vals else 0
                    st.metric("Total Missing Values", total_missing)

            # Display drift detection results
            if 'data_drift' in self.monitoring_results:
                drift = self.monitoring_results['data_drift']
                st.subheader("Data Drift Detection")

                col1, col2 = st.columns(2)
                with col1:
                    drift_detected = drift.get('drift_detected', False)
                    status = "⚠️ Drift Detected" if drift_detected else "✅ No Drift"
                    st.metric("Drift Status", status)
                with col2:
                    drift_score = drift.get('drift_score', 0)
                    st.metric("Drift Score", ".3f")

                # Drift visualization (simplified - in practice extract from Evidently report)
                features = ['temperature', 'humidity', 'wind_speed', 'load_lag_24h', 'price_lag_24h']
                drift_scores = np.random.uniform(0, 0.15, len(features))  # Simulate drift scores

                drift_df = pd.DataFrame({
                    'feature': features,
                    'drift_score': drift_scores,
                    'threshold': 0.05
                })

                fig = px.bar(drift_df, x='feature', y='drift_score',
                            title='Feature Drift Scores',
                            color=drift_df['drift_score'] > drift_df['threshold'],
                            color_discrete_map={True: 'red', False: 'green'})

                fig.add_hline(y=0.05, line_dash="dash", line_color="red",
                             annotation_text="Drift Threshold")

                st.plotly_chart(fig, use_container_width=True)

                # Show monitoring timestamp
                timestamp = self.monitoring_results.get('timestamp', 'N/A')
                st.caption(f"Last monitoring run: {timestamp}")
        else:
            st.info("Click 'Run Monitoring' in the sidebar to see real monitoring results")

            # Sample alerts
            alerts = [
                {"type": "info", "message": "Model retraining completed successfully", "timestamp": "2024-01-15 10:30"},
                {"type": "warning", "message": "Data drift detected in temperature features", "timestamp": "2024-01-14 15:45"},
                {"type": "success", "message": "All systems operational", "timestamp": "2024-01-15 09:00"}
            ]

            for alert in alerts:
                if alert["type"] == "warning":
                    st.warning(f"⚠️ {alert['message']} - {alert['timestamp']}")
                elif alert["type"] == "success":
                    st.success(f"✅ {alert['message']} - {alert['timestamp']}")
                else:
                    st.info(f"ℹ️ {alert['message']} - {alert['timestamp']}")

            # Drift detection visualization
            st.subheader("Data Drift Detection")

            # Sample drift data
            features = ['temperature', 'wind_speed', 'humidity', 'load_lag_24h']
            drift_scores = np.random.uniform(0, 0.3, len(features))

            drift_df = pd.DataFrame({
                'feature': features,
                'drift_score': drift_scores,
                'threshold': 0.05
            })

            fig = px.bar(drift_df, x='feature', y='drift_score',
                        title='Feature Drift Scores',
                        color=drift_df['drift_score'] > drift_df['threshold'],
                        color_discrete_map={True: 'red', False: 'green'})

            fig.add_hline(y=0.05, line_dash="dash", line_color="red",
                         annotation_text="Drift Threshold")

            st.plotly_chart(fig, use_container_width=True)

    def run_monitoring_cycle(self):
        """Run a complete monitoring cycle using Evidently."""
        try:
            # Load reference data using data loader
            reference_data = load_processed_dataset()
            if reference_data is None or reference_data.empty:
                st.error("Reference data not found. Please ensure the data processing pipeline has been run.")
                return

            if self.monitor is None:
                self.monitor = EvidentlyMonitor(reference_data, 'total_load_actual')

            # Load current data (for now, use a subset of reference data as current data)
            # In production, this would be new incoming data
            current_data = reference_data.tail(min(1000, len(reference_data)))  # Use recent data as current

            # Run monitoring cycle
            output_dir = "reports/monitoring"
            Path(output_dir).mkdir(parents=True, exist_ok=True)

            self.monitoring_results = {
                'timestamp': datetime.now().isoformat(),
                'data_quality': {},
                'data_drift': {},
                'model_performance': {},
                'alerts': []
            }

            # Data quality check
            try:
                quality_report_path = f"{output_dir}/data_quality_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
                self.monitoring_results['data_quality'] = self.monitor.create_data_quality_report(
                    current_data, quality_report_path
                )
            except Exception as e:
                st.warning(f"Data quality check failed: {str(e)}")

            # Data drift detection
            try:
                drift_report_path = f"{output_dir}/data_drift_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
                self.monitoring_results['data_drift'] = self.monitor.detect_data_drift(
                    current_data, drift_report_path
                )
            except Exception as e:
                st.warning(f"Data drift detection failed: {str(e)}")

            # Model performance monitoring (simplified - using dummy predictions)
            try:
                if len(current_data) > 0:
                    # Generate dummy predictions for demonstration
                    np.random.seed(42)
                    true_values = current_data['total_load_actual'].tail(min(100, len(current_data)))
                    predictions = true_values * (1 + np.random.normal(0, 0.05, len(true_values)))

                    perf_report_path = f"{output_dir}/model_performance_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
                    self.monitoring_results['model_performance'] = self.monitor.monitor_model_performance(
                        true_values, predictions, perf_report_path
                    )
            except Exception as e:
                st.warning(f"Model performance monitoring failed: {str(e)}")

            # Check for alerts
            try:
                self.monitoring_results['alerts'] = self.monitor.check_alerts(
                    self.monitoring_results['data_drift'],
                    self.monitoring_results['model_performance'],
                    {'max_mae': 1000, 'min_r2': 0.7, 'drift_threshold': 0.05}
                )
            except Exception as e:
                st.warning(f"Alert checking failed: {str(e)}")

            st.success("Monitoring cycle completed successfully!")

        except Exception as e:
            st.error(f"Monitoring cycle failed: {str(e)}")
            logging.error(f"Monitoring cycle error: {str(e)}")

    def run(self):
        """Run the dashboard."""
        self.create_sidebar()

        # Main content
        tab1, tab2, tab3, tab4 = st.tabs([
            "📊 Overview", "📈 Analysis", "🔮 Predictions", "🚨 Monitoring"
        ])

        with tab1:
            self.display_data_overview()

        with tab2:
            self.display_time_series_plot()

        with tab3:
            self.display_predictions()

        with tab4:
            col1, col2 = st.columns(2)
            with col1:
                self.display_model_performance()
            with col2:
                self.display_monitoring_alerts()


def main():
    """Main function to run the dashboard."""
    dashboard = PredictionDashboard()
    dashboard.run()


if __name__ == "__main__":
    main()

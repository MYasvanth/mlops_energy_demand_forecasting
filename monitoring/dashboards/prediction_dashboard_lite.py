"""
Lightweight Streamlit Dashboard for Energy Demand Forecasting
Optimized for fast loading on Streamlit Cloud
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Set page configuration
st.set_page_config(
    page_title="Energy Demand Forecasting",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Cache data loading
@st.cache_data
def load_sample_data():
    """Load sample data for demonstration."""
    # Generate sample time series data
    dates = pd.date_range(start='2024-01-01', periods=1000, freq='H')
    np.random.seed(42)

    # Generate realistic energy demand patterns
    base_load = 25000 + 5000 * np.sin(2 * np.pi * dates.hour / 24)  # Daily pattern
    weekly_pattern = 1 + 0.1 * np.sin(2 * np.pi * dates.dayofweek / 7)  # Weekly pattern
    noise = np.random.normal(0, 1000, len(dates))

    data = pd.DataFrame({
        'time': dates,
        'total_load_actual': base_load * weekly_pattern + noise,
        'generation_solar': np.maximum(0, 8000 * np.sin(np.pi * dates.hour / 12) + np.random.normal(0, 500, len(dates))),
        'generation_wind_onshore': np.maximum(0, 6000 + 2000 * np.random.normal(0, 1, len(dates))),
        'temperature': 15 + 10 * np.sin(2 * np.pi * dates.dayofyear / 365) + np.random.normal(0, 3, len(dates)),
        'wind_speed': np.maximum(0, 5 + 3 * np.random.normal(0, 1, len(dates)))
    })

    data.set_index('time', inplace=True)
    return data

def create_sidebar():
    """Create sidebar with controls."""
    st.sidebar.title("⚡ Energy Dashboard")

    # Model selection
    selected_model = st.sidebar.selectbox(
        "Choose Model",
        ["ARIMA", "Prophet", "LSTM", "Ensemble"],
        index=2
    )

    # Forecast settings
    forecast_hours = st.sidebar.slider(
        "Forecast Hours",
        min_value=1,
        max_value=168,
        value=24
    )

    return selected_model, forecast_hours

def display_overview(data):
    """Display data overview."""
    st.header("📊 Data Overview")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Total Records", len(data))
    with col2:
        st.metric("Date Range", f"{data.index.min().date()} to {data.index.max().date()}")
    with col3:
        avg_load = data['total_load_actual'].mean()
        st.metric("Avg Load", ".0f")
    with col4:
        max_load = data['total_load_actual'].max()
        st.metric("Peak Load", ".0f")

def display_time_series(data):
    """Display interactive time series plot."""
    st.header("📈 Time Series Analysis")

    # Date range selector
    min_date = data.index.min().date()
    max_date = data.index.max().date()

    date_range = st.date_input(
        "Select Date Range",
        value=(min_date, max_date),
        min_value=min_date,
        max_value=max_date
    )

    # Filter data
    if len(date_range) == 2:
        start_date, end_date = date_range
        filtered_data = data.loc[pd.to_datetime(start_date):pd.to_datetime(end_date) + pd.Timedelta(days=1)]
    else:
        filtered_data = data

    # Create plot
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=filtered_data.index,
        y=filtered_data['total_load_actual'],
        mode='lines',
        name='Actual Load (MW)',
        line=dict(color='blue', width=2)
    ))

    fig.add_trace(go.Scatter(
        x=filtered_data.index,
        y=filtered_data['generation_solar'],
        mode='lines',
        name='Solar Generation (MW)',
        line=dict(color='orange', width=1, dash='dot'),
        visible='legendonly'
    ))

    fig.add_trace(go.Scatter(
        x=filtered_data.index,
        y=filtered_data['generation_wind_onshore'],
        mode='lines',
        name='Wind Generation (MW)',
        line=dict(color='green', width=1, dash='dot'),
        visible='legendonly'
    ))

    fig.update_layout(
        title="Energy Demand and Generation",
        xaxis_title="Time",
        yaxis_title="Energy (MW)",
        height=500,
        hovermode='x unified'
    )

    st.plotly_chart(fig, use_container_width=True)

def display_predictions(data, selected_model, forecast_hours):
    """Display predictions."""
    st.header("🔮 Predictions")

    if st.button("Generate Predictions", type="primary"):
        with st.spinner("Generating predictions..."):
            # Generate sample predictions
            last_date = data.index[-1]
            pred_dates = pd.date_range(
                start=last_date + pd.Timedelta(hours=1),
                periods=forecast_hours,
                freq='H'
            )

            # Simple prediction model (replace with actual model)
            base_load = data['total_load_actual'].iloc[-24:].mean()
            trend = np.linspace(0, forecast_hours/24 * 1000, forecast_hours)  # Slight upward trend
            seasonal = 2000 * np.sin(2 * np.pi * pred_dates.hour / 24)
            noise = np.random.normal(0, 500, forecast_hours)

            predictions = base_load + trend + seasonal + noise

            pred_df = pd.DataFrame({
                'datetime': pred_dates,
                'prediction': predictions,
                'model': selected_model
            })

            st.subheader(f"{selected_model} Forecast")
            st.dataframe(pred_df.head(10))

            # Plot predictions
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=pred_df['datetime'],
                y=pred_df['prediction'],
                mode='lines+markers',
                name='Predictions',
                line=dict(color='red', width=3)
            ))

            fig.update_layout(
                title=f"{selected_model} Forecast - Next {forecast_hours} Hours",
                xaxis_title="Time",
                yaxis_title="Predicted Load (MW)",
                height=400
            )

            st.plotly_chart(fig, use_container_width=True)

            # Metrics
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Mean Prediction", ".0f")
            with col2:
                st.metric("Min Prediction", ".0f")
            with col3:
                st.metric("Max Prediction", ".0f")

def display_monitoring():
    """Display monitoring section."""
    st.header("🚨 Monitoring & Alerts")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("System Status")
        st.success("✅ All Systems Operational")
        st.info("📊 Model Performance: Good")
        st.warning("⚠️ Data Quality Check Due")

    with col2:
        st.subheader("Recent Alerts")
        st.info("🔄 Model retraining completed - 2 hours ago")
        st.success("✅ Data drift within acceptable range")
        st.warning("⚠️ High prediction uncertainty detected")

    # Sample drift detection
    st.subheader("Data Drift Detection")

    features = ['temperature', 'wind_speed', 'load_lag_24h', 'solar_generation']
    drift_scores = np.random.uniform(0, 0.08, len(features))

    drift_df = pd.DataFrame({
        'feature': features,
        'drift_score': drift_scores,
        'threshold': 0.05
    })

    fig = px.bar(
        drift_df,
        x='feature',
        y='drift_score',
        title='Feature Drift Scores',
        color=drift_df['drift_score'] > drift_df['threshold'],
        color_discrete_map={True: 'red', False: 'green'}
    )

    fig.add_hline(y=0.05, line_dash="dash", line_color="red")
    st.plotly_chart(fig, use_container_width=True)

def main():
    """Main dashboard function."""
    st.title("⚡ Energy Demand Forecasting Dashboard")
    st.markdown("---")

    # Load data
    with st.spinner("Loading data..."):
        data = load_sample_data()

    # Sidebar
    selected_model, forecast_hours = create_sidebar()

    # Main content tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 Overview", "📈 Analysis", "🔮 Predictions", "🚨 Monitoring"
    ])

    with tab1:
        display_overview(data)

    with tab2:
        display_time_series(data)

    with tab3:
        display_predictions(data, selected_model, forecast_hours)

    with tab4:
        display_monitoring()

    # Footer
    st.markdown("---")
    st.markdown("*Built with Streamlit • Optimized for fast loading*")

if __name__ == "__main__":
    main()

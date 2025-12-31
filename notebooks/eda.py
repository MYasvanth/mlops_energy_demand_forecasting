"""
Exploratory Data Analysis (EDA) for Energy Demand Forecasting

This script performs comprehensive exploratory data analysis on the energy and weather datasets.
It includes statistical summaries, time series analysis, correlation analysis, outlier detection,
and cross-domain relationships with proper separation of concerns.
"""

import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path
import sys
from typing import Dict, List, Tuple, Optional
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller, acf, pacf
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler

# Add src to path for importing modules
sys.path.append(str(Path(__file__).parent.parent / 'src'))

from data.ingestion import ingest_data

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set style for plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")


def load_data() -> dict:
    """
    Load energy and weather data using the ingestion module.

    Returns:
        dict: Dictionary containing 'energy' and 'weather' DataFrames.
    """
    energy_path = Path(__file__).parent.parent / 'data' / 'raw' / 'energy_dataset.csv'
    weather_path = Path(__file__).parent.parent / 'data' / 'raw' / 'weather_features.csv'

    logger.info("Loading data for EDA...")
    data = ingest_data(str(energy_path), str(weather_path))
    return data


def basic_statistics(df: pd.DataFrame, name: str) -> None:
    """
    Print basic statistics for the DataFrame.

    Args:
        df (pd.DataFrame): DataFrame to analyze.
        name (str): Name of the dataset for logging.
    """
    logger.info(f"Basic statistics for {name} dataset:")
    print(f"\n{name} Dataset Shape: {df.shape}")
    print(f"\n{name} Data Types:\n{df.dtypes}")
    print(f"\n{name} Summary Statistics:\n{df.describe()}")
    print(f"\n{name} Missing Values:\n{df.isnull().sum()}")


def plot_time_series(df: pd.DataFrame, columns: list, title: str) -> None:
    """
    Plot time series for selected columns.

    Args:
        df (pd.DataFrame): DataFrame with time series data.
        columns (list): List of columns to plot.
        title (str): Title for the plot.
    """
    df_copy = df.copy()
    df_copy['time'] = pd.to_datetime(df_copy['time'])
    df_copy.set_index('time', inplace=True)

    plt.figure(figsize=(15, 8))
    for col in columns:
        if col in df_copy.columns:
            plt.plot(df_copy.index, df_copy[col], label=col, alpha=0.7)

    plt.title(title)
    plt.xlabel('Time')
    plt.ylabel('Value')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.show()


def correlation_heatmap(df: pd.DataFrame, title: str) -> None:
    """
    Plot correlation heatmap for numerical columns.

    Args:
        df (pd.DataFrame): DataFrame to analyze.
        title (str): Title for the plot.
    """
    numerical_cols = df.select_dtypes(include=[np.number]).columns
    corr_matrix = df[numerical_cols].corr()

    plt.figure(figsize=(12, 10))
    sns.heatmap(corr_matrix, annot=False, cmap='coolwarm', center=0, linewidths=0.5)
    plt.title(title)
    plt.tight_layout()
    plt.show()


def distribution_plots(df: pd.DataFrame, columns: list, title: str) -> None:
    """
    Plot distribution histograms for selected columns.

    Args:
        df (pd.DataFrame): DataFrame to analyze.
        columns (list): List of columns to plot.
        title (str): Title for the plot.
    """
    num_cols = len(columns)
    fig, axes = plt.subplots(nrows=(num_cols + 2) // 3, ncols=3, figsize=(15, 5 * ((num_cols + 2) // 3)))
    axes = axes.flatten()

    for i, col in enumerate(columns):
        if col in df.columns:
            sns.histplot(df[col].dropna(), ax=axes[i], kde=True)
            axes[i].set_title(f'Distribution of {col}')
        else:
            axes[i].set_visible(False)

    plt.suptitle(title)
    plt.tight_layout()
    plt.show()


def box_plots(df: pd.DataFrame, columns: list, title: str) -> None:
    """
    Plot box plots for selected columns to detect outliers.

    Args:
        df (pd.DataFrame): DataFrame to analyze.
        columns (list): List of columns to plot.
        title (str): Title for the plot.
    """
    plt.figure(figsize=(15, 8))
    df_melted = df[columns].melt(var_name='Variable', value_name='Value')
    sns.boxplot(x='Variable', y='Value', data=df_melted)
    plt.title(title)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


def weather_analysis(weather_df: pd.DataFrame) -> None:
    """
    Perform specific analysis on weather data.

    Args:
        weather_df (pd.DataFrame): Weather DataFrame.
    """
    logger.info("Analyzing weather data...")

    # Convert dt_iso to datetime
    weather_df['dt_iso'] = pd.to_datetime(weather_df['dt_iso'])

    # Temperature over time
    plt.figure(figsize=(15, 6))
    plt.plot(weather_df['dt_iso'], weather_df['temp'], label='Temperature')
    plt.plot(weather_df['dt_iso'], weather_df['temp_min'], label='Min Temp', alpha=0.7)
    plt.plot(weather_df['dt_iso'], weather_df['temp_max'], label='Max Temp', alpha=0.7)
    plt.title('Weather Temperature Over Time')
    plt.xlabel('Time')
    plt.ylabel('Temperature (°C)')
    plt.legend()
    plt.show()

    # Humidity and pressure
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10))
    ax1.plot(weather_df['dt_iso'], weather_df['humidity'], color='blue')
    ax1.set_title('Humidity Over Time')
    ax1.set_ylabel('Humidity (%)')

    ax2.plot(weather_df['dt_iso'], weather_df['pressure'], color='green')
    ax2.set_title('Pressure Over Time')
    ax2.set_xlabel('Time')
    ax2.set_ylabel('Pressure (hPa)')
    plt.tight_layout()
    plt.show()


class EDAAnalyzer:
    """
    Comprehensive EDA analyzer with separation of concerns.
    Each method focuses on a specific aspect of data analysis.
    """

    def __init__(self, data_dict: Dict[str, pd.DataFrame]):
        """
        Initialize EDA analyzer with data.

        Args:
            data_dict: Dictionary containing 'energy' and 'weather' DataFrames
        """
        self.energy_df = data_dict['energy'].copy()
        self.weather_df = data_dict['weather'].copy()
        self.merged_df = None

    def perform_data_overview(self) -> None:
        """Perform basic data overview and statistics."""
        logger.info("=== DATA OVERVIEW ANALYSIS ===")

        # Basic statistics
        basic_statistics(self.energy_df, "Energy")
        basic_statistics(self.weather_df, "Weather")

        # Data quality assessment
        self._analyze_data_quality()

    def perform_univariate_analysis(self) -> None:
        """Perform univariate analysis on key variables."""
        logger.info("=== UNIVARIATE ANALYSIS ===")

        energy_columns = ['total_load_actual', 'price_actual', 'generation_solar', 'generation_wind_onshore']

        # Distribution plots
        distribution_plots(self.energy_df, energy_columns, 'Energy Data Distributions')

        # Box plots for outlier detection
        box_plots(self.energy_df, energy_columns, 'Energy Data Box Plots - Outlier Detection')

    def perform_time_series_analysis(self) -> None:
        """Perform comprehensive time series analysis."""
        logger.info("=== TIME SERIES ANALYSIS ===")

        # Basic time series plots
        energy_columns = ['total_load_actual', 'price_actual', 'generation_solar', 'generation_wind_onshore']
        plot_time_series(self.energy_df, energy_columns, 'Energy Generation and Load Over Time')

        # Advanced time series analysis
        self._perform_advanced_ts_analysis()

    def perform_correlation_analysis(self) -> None:
        """Perform correlation analysis within and across datasets."""
        logger.info("=== CORRELATION ANALYSIS ===")

        # Within-dataset correlations
        correlation_heatmap(self.energy_df, 'Energy Data Correlation Heatmap')

        # Cross-dataset analysis
        self._perform_cross_dataset_analysis()

    def perform_weather_analysis(self) -> None:
        """Perform dedicated weather data analysis."""
        logger.info("=== WEATHER DATA ANALYSIS ===")
        weather_analysis(self.weather_df)

    def perform_feature_importance_analysis(self) -> None:
        """Perform preliminary feature importance analysis."""
        logger.info("=== FEATURE IMPORTANCE ANALYSIS ===")

        # Prepare data for analysis
        self._prepare_merged_data()

        if self.merged_df is not None:
            self._analyze_feature_importance()

    def _analyze_data_quality(self) -> None:
        """Analyze data quality metrics."""
        print("\n=== DATA QUALITY INSIGHTS ===")

        # Missing value patterns
        energy_missing = self.energy_df.isnull().sum()
        weather_missing = self.weather_df.isnull().sum()

        print(f"Energy data - Total missing values: {energy_missing.sum()}")
        print(f"Weather data - Total missing values: {weather_missing.sum()}")

        # Outlier analysis using IQR
        self._detect_outliers_iqr()

    def _detect_outliers_iqr(self) -> None:
        """Detect outliers using IQR method."""
        target_col = 'total_load_actual'
        if target_col in self.energy_df.columns:
            Q1 = self.energy_df[target_col].quantile(0.25)
            Q3 = self.energy_df[target_col].quantile(0.75)
            IQR = Q3 - Q1
            outliers = self.energy_df[
                (self.energy_df[target_col] < (Q1 - 1.5 * IQR)) |
                (self.energy_df[target_col] > (Q3 + 1.5 * IQR))
            ]
            print(f"Outliers in {target_col}: {len(outliers)} records ({len(outliers)/len(self.energy_df)*100:.1f}%)")

    def _perform_advanced_ts_analysis(self) -> None:
        """Perform advanced time series analysis."""
        print("\n=== ADVANCED TIME SERIES ANALYSIS ===")

        target_col = 'total_load_actual'
        if target_col in self.energy_df.columns:
            # Stationarity test
            self._test_stationarity(target_col)

            # Seasonal decomposition
            self._seasonal_decomposition(target_col)

            # Autocorrelation analysis
            self._autocorrelation_analysis(target_col)

    def _test_stationarity(self, column: str) -> None:
        """Test for stationarity using ADF test."""
        try:
            result = adfuller(self.energy_df[column].dropna())
            print(f"ADF Test for {column}:")
            print(f"  Test Statistic: {result[0]:.4f}")
            print(f"  p-value: {result[1]:.4f}")
            print(f"  Stationary: {'Yes' if result[1] < 0.05 else 'No'}")
        except Exception as e:
            logger.warning(f"Could not perform stationarity test: {e}")

    def _seasonal_decomposition(self, column: str) -> None:
        """Perform seasonal decomposition."""
        try:
            # Set datetime index
            temp_df = self.energy_df.copy()
            temp_df['time'] = pd.to_datetime(temp_df['time'])
            temp_df = temp_df.set_index('time')[column].dropna()

            # Perform decomposition (daily seasonality)
            decomposition = seasonal_decompose(temp_df, model='additive', period=24)
            print(f"Seasonal decomposition completed for {column}")

            # Plot decomposition
            fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(15, 12))
            decomposition.observed.plot(ax=ax1, title='Observed')
            decomposition.trend.plot(ax=ax2, title='Trend')
            decomposition.seasonal.plot(ax=ax3, title='Seasonal')
            decomposition.resid.plot(ax=ax4, title='Residual')
            plt.tight_layout()
            plt.show()

        except Exception as e:
            logger.warning(f"Could not perform seasonal decomposition: {e}")

    def _autocorrelation_analysis(self, column: str) -> None:
        """Perform autocorrelation analysis."""
        try:
            # Calculate ACF and PACF
            data = self.energy_df[column].dropna()
            lag_acf = acf(data, nlags=50)
            lag_pacf = pacf(data, nlags=50)

            # Plot ACF and PACF
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10))
            plot_acf(data, lags=50, ax=ax1, title=f'Autocorrelation Function - {column}')
            plot_pacf(data, lags=50, ax=ax2, title=f'Partial Autocorrelation Function - {column}')
            plt.tight_layout()
            plt.show()

        except Exception as e:
            logger.warning(f"Could not perform autocorrelation analysis: {e}")

    def _perform_cross_dataset_analysis(self) -> None:
        """Analyze relationships between energy and weather data."""
        print("\n=== CROSS-DOMAIN ANALYSIS ===")

        try:
            # Merge datasets on time
            self._prepare_merged_data()

            if self.merged_df is not None:
                # Energy-Weather correlations
                key_vars = ['total_load_actual', 'temp', 'humidity', 'wind_speed', 'pressure']
                available_vars = [col for col in key_vars if col in self.merged_df.columns]

                if len(available_vars) > 1:
                    corr_matrix = self.merged_df[available_vars].corr()
                    print("Energy-Weather Correlation Matrix:")
                    print(corr_matrix)

                    # Visualize cross-domain correlations
                    plt.figure(figsize=(10, 8))
                    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, linewidths=0.5)
                    plt.title('Energy-Weather Cross-Domain Correlations')
                    plt.tight_layout()
                    plt.show()

                # Lag correlation analysis
                self._analyze_lag_correlations()

        except Exception as e:
            logger.warning(f"Could not perform cross-dataset analysis: {e}")

    def _prepare_merged_data(self) -> None:
        """Prepare merged energy-weather dataset."""
        try:
            # Convert time columns to datetime with UTC handling
            energy_temp = self.energy_df.copy()
            weather_temp = self.weather_df.copy()

            energy_temp['time'] = pd.to_datetime(energy_temp['time'], utc=True)
            weather_temp['dt_iso'] = pd.to_datetime(weather_temp['dt_iso'], utc=True)

            # Merge on time (assuming hourly alignment)
            self.merged_df = pd.merge(
                energy_temp, weather_temp,
                left_on='time', right_on='dt_iso',
                how='inner'
            )
            print(f"Merged dataset shape: {self.merged_df.shape}")

        except Exception as e:
            logger.warning(f"Could not merge datasets: {e}")
            self.merged_df = None

    def _analyze_lag_correlations(self) -> None:
        """Analyze lag correlations between energy and weather."""
        if self.merged_df is None:
            return

        try:
            target = 'total_load_actual'
            weather_vars = ['temp', 'humidity', 'wind_speed']

            print("\nLag Correlation Analysis (Energy Load vs Weather):")
            for weather_var in weather_vars:
                if weather_var in self.merged_df.columns and target in self.merged_df.columns:
                    max_corr = 0
                    best_lag = 0

                    # Test different lags
                    for lag in range(0, 25):  # 0-24 hour lags
                        if lag == 0:
                            corr = self.merged_df[target].corr(self.merged_df[weather_var])
                        else:
                            corr = self.merged_df[target].corr(self.merged_df[weather_var].shift(lag))

                        if abs(corr) > abs(max_corr):
                            max_corr = corr
                            best_lag = lag

                    print(f"  {weather_var}: Best correlation = {max_corr:.3f} at lag {best_lag}h")

        except Exception as e:
            logger.warning(f"Could not analyze lag correlations: {e}")

    def _analyze_feature_importance(self) -> None:
        """Perform preliminary feature importance analysis."""
        if self.merged_df is None:
            return

        try:
            # Prepare features and target
            target = 'total_load_actual'
            feature_cols = [col for col in self.merged_df.select_dtypes(include=[np.number]).columns
                          if col != target and not col.endswith('_target_')]

            # Remove target features to avoid data leakage
            feature_cols = [col for col in feature_cols if '_target_' not in col]

            if len(feature_cols) > 5:  # Need minimum features
                X = self.merged_df[feature_cols].fillna(self.merged_df[feature_cols].mean())
                y = self.merged_df[target]

                # Scale features
                scaler = StandardScaler()
                X_scaled = scaler.fit_transform(X)

                # Train simple model for feature importance
                model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
                model.fit(X_scaled, y)

                # Get feature importance
                importance_df = pd.DataFrame({
                    'feature': feature_cols,
                    'importance': model.feature_importances_
                }).sort_values('importance', ascending=False)

                print("\nTop 10 Most Important Features:")
                print(importance_df.head(10))

                # Visualize feature importance
                plt.figure(figsize=(12, 8))
                top_features = importance_df.head(15)
                sns.barplot(x='importance', y='feature', data=top_features)
                plt.title('Top 15 Feature Importance (Random Forest)')
                plt.xlabel('Importance Score')
                plt.ylabel('Features')
                plt.tight_layout()
                plt.show()

        except Exception as e:
            logger.warning(f"Could not perform feature importance analysis: {e}")

    def run_complete_analysis(self) -> None:
        """Run complete EDA analysis with all components."""
        logger.info("Starting Comprehensive Exploratory Data Analysis...")

        # Execute analysis in logical order
        self.perform_data_overview()
        self.perform_univariate_analysis()
        self.perform_time_series_analysis()
        self.perform_weather_analysis()
        self.perform_correlation_analysis()
        self.perform_feature_importance_analysis()

        logger.info("Comprehensive EDA completed. Check generated plots and insights.")


def main():
    """
    Main function to run comprehensive EDA.
    """
    # Load data
    data = load_data()

    # Create EDA analyzer
    analyzer = EDAAnalyzer(data)

    # Run complete analysis
    analyzer.run_complete_analysis()


if __name__ == "__main__":
    main()

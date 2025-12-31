"""
Custom Business Metrics for Energy Demand Forecasting

This module implements energy-specific business metrics and KPIs
for monitoring model performance and business impact.
"""

import logging
from typing import Dict, List, Optional, Tuple, Any
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path

logger = logging.getLogger(__name__)


class EnergyBusinessMetrics:
    """
    Custom business metrics for energy demand forecasting.
    """

    def __init__(self, target_column: str = 'total_load_actual'):
        """
        Initialize business metrics calculator.

        Args:
            target_column (str): Name of the target column.
        """
        self.target_column = target_column
        self.metrics_history = []

    def calculate_peak_demand_accuracy(self, y_true: pd.Series, y_pred: pd.Series,
                                     peak_hours: List[int] = [17, 18, 19, 20]) -> Dict[str, float]:
        """
        Calculate peak demand forecasting accuracy.

        Args:
            y_true (pd.Series): True values.
            y_pred (pd.Series): Predicted values.
            peak_hours (List[int]): Hours considered as peak demand.

        Returns:
            Dict[str, float]: Peak demand metrics.
        """
        try:
            # Identify peak hours in the data
            if hasattr(y_true.index, 'hour'):
                peak_mask = y_true.index.hour.isin(peak_hours)
            else:
                # If no datetime index, assume every 4th point is peak (simplified)
                peak_mask = np.arange(len(y_true)) % 6 >= 3  # Rough approximation

            if peak_mask.sum() == 0:
                logger.warning("No peak hours identified in data")
                return {
                    'peak_mae': 0.0,
                    'peak_mape': 0.0,
                    'peak_accuracy': 0.0,
                    'peak_hours_count': 0
                }

            y_true_peak = y_true[peak_mask]
            y_pred_peak = y_pred[peak_mask]

            # Calculate peak-specific metrics
            peak_mae = np.mean(np.abs(y_true_peak - y_pred_peak))
            peak_mape = np.mean(np.abs((y_true_peak - y_pred_peak) / y_true_peak)) * 100
            peak_accuracy = 100 - peak_mape

            return {
                'peak_mae': peak_mae,
                'peak_mape': peak_mape,
                'peak_accuracy': peak_accuracy,
                'peak_hours_count': len(y_true_peak)
            }

        except Exception as e:
            logger.error(f"Error calculating peak demand accuracy: {str(e)}")
            return {
                'peak_mae': 0.0,
                'peak_mape': 0.0,
                'peak_accuracy': 0.0,
                'peak_hours_count': 0
            }

    def calculate_renewable_contribution_tracking(self, data: pd.DataFrame,
                                                renewable_cols: List[str] = None) -> Dict[str, float]:
        """
        Track renewable energy contribution and forecast accuracy.

        Args:
            data (pd.DataFrame): Dataset with renewable generation columns.
            renewable_cols (List[str]): Renewable energy columns.

        Returns:
            Dict[str, float]: Renewable energy metrics.
        """
        if renewable_cols is None:
            renewable_cols = ['generation_solar', 'generation_wind_onshore',
                            'generation_hydro_water_reservoir']

        try:
            renewable_metrics = {}

            # Calculate total renewable generation
            available_cols = [col for col in renewable_cols if col in data.columns]
            if available_cols:
                data['total_renewable'] = data[available_cols].sum(axis=1)
                renewable_metrics['avg_renewable_generation'] = data['total_renewable'].mean()
                renewable_metrics['max_renewable_generation'] = data['total_renewable'].max()
                renewable_metrics['renewable_penetration'] = (
                    data['total_renewable'].sum() / data[self.target_column].sum()
                ) * 100

                # Renewable forecast accuracy (if forecast columns exist)
                forecast_cols = [f"{col}_forecast" for col in available_cols]
                available_forecast_cols = [col for col in forecast_cols if col in data.columns]

                if available_forecast_cols:
                    renewable_true = data[available_cols].sum(axis=1)
                    renewable_pred = data[available_forecast_cols].sum(axis=1)
                    renewable_metrics['renewable_forecast_mae'] = np.mean(np.abs(renewable_true - renewable_pred))
                    renewable_metrics['renewable_forecast_mape'] = (
                        np.mean(np.abs((renewable_true - renewable_pred) / renewable_true)) * 100
                    )
            else:
                renewable_metrics.update({
                    'avg_renewable_generation': 0.0,
                    'max_renewable_generation': 0.0,
                    'renewable_penetration': 0.0,
                    'renewable_forecast_mae': 0.0,
                    'renewable_forecast_mape': 0.0
                })

            return renewable_metrics

        except Exception as e:
            logger.error(f"Error calculating renewable contribution: {str(e)}")
            return {
                'avg_renewable_generation': 0.0,
                'max_renewable_generation': 0.0,
                'renewable_penetration': 0.0,
                'renewable_forecast_mae': 0.0,
                'renewable_forecast_mape': 0.0
            }

    def calculate_forecast_reliability(self, y_true: pd.Series, y_pred: pd.Series,
                                     confidence_intervals: Optional[Tuple[float, float]] = None) -> Dict[str, float]:
        """
        Calculate forecast reliability and confidence metrics.

        Args:
            y_true (pd.Series): True values.
            y_pred (pd.Series): Predicted values.
            confidence_intervals (Optional[Tuple[float, float]]): Prediction intervals.

        Returns:
            Dict[str, float]: Reliability metrics.
        """
        try:
            # Basic reliability metrics
            errors = y_true - y_pred
            reliability_metrics = {
                'forecast_bias': errors.mean(),
                'forecast_variance': errors.var(),
                'forecast_consistency': 1 / (1 + errors.std()),  # Inverse of error std
                'prediction_interval_coverage': 0.0  # Will be calculated if intervals provided
            }

            # Calculate prediction interval coverage if available
            if confidence_intervals is not None:
                lower_bound, upper_bound = confidence_intervals
                coverage = np.mean((y_pred >= lower_bound) & (y_pred <= upper_bound))
                reliability_metrics['prediction_interval_coverage'] = coverage * 100

            # Reliability score (composite metric)
            reliability_score = (
                (1 - abs(reliability_metrics['forecast_bias']) / y_true.mean()) * 0.4 +
                reliability_metrics['forecast_consistency'] * 0.4 +
                min(reliability_metrics['prediction_interval_coverage'] / 95, 1.0) * 0.2
            ) * 100

            reliability_metrics['reliability_score'] = max(0, min(100, reliability_score))

            return reliability_metrics

        except Exception as e:
            logger.error(f"Error calculating forecast reliability: {str(e)}")
            return {
                'forecast_bias': 0.0,
                'forecast_variance': 0.0,
                'forecast_consistency': 0.0,
                'prediction_interval_coverage': 0.0,
                'reliability_score': 0.0
            }

    def calculate_business_impact(self, y_true: pd.Series, y_pred: pd.Series,
                                energy_price: float = 0.12) -> Dict[str, float]:
        """
        Calculate business impact of forecasting errors.

        Args:
            y_true (pd.Series): True values.
            y_pred (pd.Series): Predicted values.
            energy_price (float): Energy price per unit.

        Returns:
            Dict[str, float]: Business impact metrics.
        """
        try:
            errors = y_true - y_pred

            # Calculate monetary impact
            abs_errors = np.abs(errors)
            total_error_cost = abs_errors.sum() * energy_price

            # Under-forecast impact (opportunity cost of not preparing for high demand)
            under_forecast = errors[errors > 0]
            under_forecast_cost = under_forecast.sum() * energy_price * 1.5  # Penalty for under-preparation

            # Over-forecast impact (cost of unnecessary capacity)
            over_forecast = -errors[errors < 0]
            over_forecast_cost = over_forecast.sum() * energy_price * 0.3  # Cost of over-capacity

            # Business impact score
            max_possible_error = y_true.sum() * energy_price
            business_impact_score = (1 - total_error_cost / max_possible_error) * 100

            return {
                'total_error_cost': total_error_cost,
                'under_forecast_cost': under_forecast_cost,
                'over_forecast_cost': over_forecast_cost,
                'business_impact_score': max(0, business_impact_score),
                'error_cost_per_mwh': total_error_cost / y_true.sum() if y_true.sum() > 0 else 0
            }

        except Exception as e:
            logger.error(f"Error calculating business impact: {str(e)}")
            return {
                'total_error_cost': 0.0,
                'under_forecast_cost': 0.0,
                'over_forecast_cost': 0.0,
                'business_impact_score': 0.0,
                'error_cost_per_mwh': 0.0
            }

    def calculate_all_metrics(self, y_true: pd.Series, y_pred: pd.Series,
                            data: pd.DataFrame = None, **kwargs) -> Dict[str, Any]:
        """
        Calculate all business metrics.

        Args:
            y_true (pd.Series): True values.
            y_pred (pd.Series): Predicted values.
            data (pd.DataFrame): Full dataset for additional metrics.
            **kwargs: Additional parameters.

        Returns:
            Dict[str, Any]: All business metrics.
        """
        metrics = {
            'timestamp': datetime.now().isoformat(),
            'peak_demand': self.calculate_peak_demand_accuracy(y_true, y_pred, **kwargs),
            'forecast_reliability': self.calculate_forecast_reliability(y_true, y_pred, **kwargs),
            'business_impact': self.calculate_business_impact(y_true, y_pred, **kwargs)
        }

        if data is not None:
            metrics['renewable_contribution'] = self.calculate_renewable_contribution_tracking(data, **kwargs)

        # Store in history
        self.metrics_history.append(metrics)

        return metrics

    def get_metrics_history(self, hours: int = 24) -> List[Dict[str, Any]]:
        """
        Get recent metrics history.

        Args:
            hours (int): Hours of history to retrieve.

        Returns:
            List[Dict[str, Any]]: Recent metrics.
        """
        if not self.metrics_history:
            return []

        # Filter by time (simplified - in practice would use actual timestamps)
        recent_count = min(len(self.metrics_history), hours)
        return self.metrics_history[-recent_count:]

    def get_metrics_summary(self) -> Dict[str, Any]:
        """
        Get summary of all metrics.

        Returns:
            Dict[str, Any]: Metrics summary.
        """
        if not self.metrics_history:
            return {}

        latest = self.metrics_history[-1]

        summary = {
            'latest_timestamp': latest['timestamp'],
            'peak_accuracy': latest['peak_demand']['peak_accuracy'],
            'reliability_score': latest['forecast_reliability']['reliability_score'],
            'business_impact_score': latest['business_impact']['business_impact_score'],
            'total_error_cost': latest['business_impact']['total_error_cost']
        }

        if 'renewable_contribution' in latest:
            summary['renewable_penetration'] = latest['renewable_contribution']['renewable_penetration']

        return summary


if __name__ == "__main__":
    # Example usage
    import numpy as np

    # Create sample data
    np.random.seed(42)
    n_samples = 100
    true_values = pd.Series(1000 + np.random.normal(0, 100, n_samples))
    pred_values = true_values + np.random.normal(0, 50, n_samples)

    # Initialize metrics calculator
    metrics_calc = EnergyBusinessMetrics()

    # Calculate all metrics
    all_metrics = metrics_calc.calculate_all_metrics(true_values, pred_values)

    print("Business Metrics Results:")
    print(f"Peak Demand Accuracy: {all_metrics['peak_demand']['peak_accuracy']:.2f}%")
    print(f"Forecast Reliability Score: {all_metrics['forecast_reliability']['reliability_score']:.2f}")
    print(f"Business Impact Score: {all_metrics['business_impact']['business_impact_score']:.2f}%")
    print(f"Total Error Cost: ${all_metrics['business_impact']['total_error_cost']:.2f}")

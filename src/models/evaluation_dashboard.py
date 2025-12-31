"""
Interactive Evaluation Dashboard for Energy Demand Forecasting

This module provides an interactive Streamlit dashboard for comprehensive
model evaluation, cross-validation results visualization, and business metrics analysis.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import json
from pathlib import Path
from typing import Dict, List, Any, Optional
import logging

logger = logging.getLogger(__name__)


class EvaluationDashboard:
    """
    Interactive dashboard for model evaluation and cross-validation results.
    """

    def __init__(self, results_path: str = 'reports/model_performance/evaluation_report.json'):
        """
        Initialize the evaluation dashboard.

        Args:
            results_path (str): Path to evaluation results JSON file.
        """
        self.results_path = results_path
        self.results = self.load_results()

    def load_results(self) -> Dict[str, Any]:
        """
        Load evaluation results from JSON file.

        Returns:
            Dict[str, Any]: Evaluation results.
        """
        try:
            if Path(self.results_path).exists():
                with open(self.results_path, 'r') as f:
                    return json.load(f)
            else:
                logger.warning(f"Results file not found: {self.results_path}")
                return {}
        except Exception as e:
            logger.error(f"Error loading results: {e}")
            return {}

    def create_model_comparison_chart(self) -> go.Figure:
        """
        Create model comparison bar chart.

        Returns:
            go.Figure: Plotly figure with model comparison.
        """
        if not self.results or 'model_comparison' not in self.results:
            return go.Figure()

        models = list(self.results['model_comparison'].keys())
        mae_scores = []
        
        for model in models:
            model_data = self.results['model_comparison'][model]
            # Extract MAE from walk_forward results
            cv_metrics = model_data.get('cv_metrics', {})
            walk_forward = cv_metrics.get('walk_forward', [])
            
            if walk_forward and len(walk_forward) > 0:
                # Calculate average MAE from all walk_forward results
                mae_values = [result.get('mae', 0) for result in walk_forward if isinstance(result, dict)]
                if mae_values:
                    avg_mae = np.mean(mae_values)
                    mae_scores.append(avg_mae)
                else:
                    mae_scores.append(float('inf'))
            else:
                mae_scores.append(float('inf'))

        # Filter out infinite values
        valid_data = [(m, s) for m, s in zip(models, mae_scores) if not np.isinf(s) and s > 0]
        if not valid_data:
            return go.Figure().add_annotation(
                text="No valid MAE data available",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False
            )

        models_filtered, mae_filtered = zip(*valid_data)

        fig = go.Figure(data=[
            go.Bar(
                x=models_filtered,
                y=mae_filtered,
                marker_color='lightblue',
                name='MAE'
            )
        ])

        fig.update_layout(
            title="Model Performance Comparison (Average MAE)",
            xaxis_title="Model",
            yaxis_title="Mean Absolute Error",
            showlegend=False
        )

        return fig

    def create_business_metrics_chart(self) -> go.Figure:
        """
        Create business metrics visualization.

        Returns:
            go.Figure: Plotly figure with business metrics.
        """
        if not self.results or 'model_comparison' not in self.results:
            return go.Figure()

        models = []
        mae_values = []
        mse_values = []
        rmse_values = []
        mape_values = []
        r2_values = []

        for model, results in self.results['model_comparison'].items():
            cv_metrics = results.get('cv_metrics', {})
            walk_forward = cv_metrics.get('walk_forward', [])
            
            if walk_forward and len(walk_forward) > 0:
                # Calculate average metrics from walk_forward results
                mae_vals = [r.get('mae', 0) for r in walk_forward if isinstance(r, dict)]
                mse_vals = [r.get('mse', 0) for r in walk_forward if isinstance(r, dict)]
                rmse_vals = [r.get('rmse', 0) for r in walk_forward if isinstance(r, dict)]
                mape_vals = [r.get('mape', 0) for r in walk_forward if isinstance(r, dict)]
                r2_vals = [r.get('r2', 0) for r in walk_forward if isinstance(r, dict)]
                
                if mae_vals:
                    models.append(model)
                    mae_values.append(np.mean(mae_vals))
                    mse_values.append(np.mean(mse_vals))
                    rmse_values.append(np.mean(rmse_vals))
                    mape_values.append(np.mean(mape_vals))
                    r2_values.append(np.mean(r2_vals))

        if not models:
            return go.Figure().add_annotation(
                text="No metrics data available",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False
            )

        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('MAE', 'RMSE', 'MAPE (%)', 'R² Score')
        )

        fig.add_trace(
            go.Bar(x=models, y=mae_values, name='MAE', marker_color='lightblue'),
            row=1, col=1
        )

        fig.add_trace(
            go.Bar(x=models, y=rmse_values, name='RMSE', marker_color='lightgreen'),
            row=1, col=2
        )

        fig.add_trace(
            go.Bar(x=models, y=mape_values, name='MAPE', marker_color='orange'),
            row=2, col=1
        )

        fig.add_trace(
            go.Bar(x=models, y=r2_values, name='R²', marker_color='red'),
            row=2, col=2
        )

        fig.update_layout(
            title="Model Performance Metrics Comparison",
            showlegend=False,
            height=600
        )

        return fig

    def create_validation_timeline_chart(self, cv_results: List[Dict]) -> go.Figure:
        """
        Create validation timeline visualization.

        Args:
            cv_results (List[Dict]): Cross-validation results over time.

        Returns:
            go.Figure: Plotly figure with validation timeline.
        """
        if not cv_results:
            return go.Figure()

        steps = list(range(len(cv_results)))
        mae_values = [r.get('mae', 0) for r in cv_results]

        fig = go.Figure(data=[
            go.Scatter(
                x=steps,
                y=mae_values,
                mode='lines+markers',
                name='MAE over time',
                line=dict(color='blue', width=2),
                marker=dict(size=6)
            )
        ])

        fig.update_layout(
            title="Validation Performance Over Time",
            xaxis_title="Validation Step",
            yaxis_title="Mean Absolute Error",
            showlegend=False
        )

        return fig

    def create_summary_stats_table(self) -> pd.DataFrame:
        """
        Create summary statistics table.

        Returns:
            pd.DataFrame: Summary statistics.
        """
        if not self.results or 'model_comparison' not in self.results:
            return pd.DataFrame()

        # Generate summary from model_comparison data
        models = list(self.results['model_comparison'].keys())
        total_models = len(models)
        
        # Calculate best model based on average MAE
        best_model = None
        best_mae = float('inf')
        all_mae_values = []
        
        for model in models:
            model_data = self.results['model_comparison'][model]
            cv_metrics = model_data.get('cv_metrics', {})
            walk_forward = cv_metrics.get('walk_forward', [])
            
            if walk_forward:
                mae_values = [r.get('mae', 0) for r in walk_forward if isinstance(r, dict)]
                if mae_values:
                    avg_mae = np.mean(mae_values)
                    all_mae_values.append(avg_mae)
                    if avg_mae < best_mae:
                        best_mae = avg_mae
                        best_model = model
        
        avg_mae = np.mean(all_mae_values) if all_mae_values else 0
        
        summary_data = {
            'Total Models Evaluated': total_models,
            'Best Model': best_model or 'N/A',
            'Best Model MAE': f"{best_mae:.2f}" if best_mae != float('inf') else 'N/A',
            'Average MAE': f"{avg_mae:.2f}" if avg_mae > 0 else 'N/A',
            'Evaluation Timestamp': self.results.get('timestamp', 'N/A')
        }
        
        return pd.DataFrame({
            'Metric': list(summary_data.keys()),
            'Value': list(summary_data.values())
        })

    def run_dashboard(self):
        """
        Run the Streamlit evaluation dashboard.
        """
        st.set_page_config(
            page_title="Energy Demand Forecasting - Model Evaluation",
            page_icon="⚡",
            layout="wide"
        )

        st.title("⚡ Energy Demand Forecasting - Model Evaluation Dashboard")
        st.markdown("---")

        # Sidebar
        st.sidebar.header("Navigation")
        page = st.sidebar.radio(
            "Select View",
            ["Overview", "Model Comparison", "Business Metrics", "Validation Details", "Reports"]
        )

        # Load latest results
        if st.sidebar.button("Refresh Results"):
            self.results = self.load_results()
            st.sidebar.success("Results refreshed!")

        # Main content
        if page == "Overview":
            self.show_overview_page()
        elif page == "Model Comparison":
            self.show_model_comparison_page()
        elif page == "Business Metrics":
            self.show_business_metrics_page()
        elif page == "Validation Details":
            self.show_validation_details_page()
        elif page == "Reports":
            self.show_reports_page()

    def show_overview_page(self):
        """
        Show overview dashboard page.
        """
        col1, col2 = st.columns([2, 1])

        with col1:
            st.subheader("Model Performance Overview")
            if self.results and 'model_comparison' in self.results:
                # Calculate best model
                models = list(self.results['model_comparison'].keys())
                best_model = None
                best_mae = float('inf')
                all_mae_values = []
                
                for model in models:
                    model_data = self.results['model_comparison'][model]
                    cv_metrics = model_data.get('cv_metrics', {})
                    walk_forward = cv_metrics.get('walk_forward', [])
                    
                    if walk_forward:
                        mae_values = [r.get('mae', 0) for r in walk_forward if isinstance(r, dict)]
                        if mae_values:
                            avg_mae = np.mean(mae_values)
                            all_mae_values.append(avg_mae)
                            if avg_mae < best_mae:
                                best_mae = avg_mae
                                best_model = model
                
                st.metric("Best Model", best_model or 'N/A')
                
                col1_1, col1_2, col1_3 = st.columns(3)
                with col1_1:
                    st.metric("Models Evaluated", len(models))
                with col1_2:
                    st.metric("Best MAE", f"{best_mae:.2f}" if best_mae != float('inf') else "N/A")
                with col1_3:
                    avg_mae = np.mean(all_mae_values) if all_mae_values else 0
                    st.metric("Average MAE", f"{avg_mae:.2f}" if avg_mae > 0 else "N/A")

            # Model comparison chart
            fig = self.create_model_comparison_chart()
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            st.subheader("Summary Statistics")
            summary_df = self.create_summary_stats_table()
            if not summary_df.empty:
                st.dataframe(summary_df, use_container_width=True)
            else:
                st.info("No summary statistics available")

    def show_model_comparison_page(self):
        """
        Show detailed model comparison page.
        """
        st.subheader("Detailed Model Comparison")

        if not self.results or 'model_comparison' not in self.results:
            st.error("No model comparison data available")
            return

        # Model selection
        models = list(self.results['model_comparison'].keys())
        selected_models = st.multiselect(
            "Select models to compare",
            models,
            default=models[:2] if len(models) >= 2 else models
        )

        if selected_models:
            # Create comparison table
            comparison_data = []
            for model in selected_models:
                results = self.results['model_comparison'][model]
                cv_metrics = results.get('cv_metrics', {})
                walk_forward = cv_metrics.get('walk_forward', [])
                
                if walk_forward:
                    mae_values = [r.get('mae', 0) for r in walk_forward if isinstance(r, dict)]
                    mse_values = [r.get('mse', 0) for r in walk_forward if isinstance(r, dict)]
                    rmse_values = [r.get('rmse', 0) for r in walk_forward if isinstance(r, dict)]
                    mape_values = [r.get('mape', 0) for r in walk_forward if isinstance(r, dict)]
                    r2_values = [r.get('r2', 0) for r in walk_forward if isinstance(r, dict)]
                    
                    comparison_data.append({
                        'Model': model,
                        'Avg MAE': f"{np.mean(mae_values):.2f}" if mae_values else 'N/A',
                        'Avg RMSE': f"{np.mean(rmse_values):.2f}" if rmse_values else 'N/A',
                        'Avg MAPE': f"{np.mean(mape_values):.2f}%" if mape_values else 'N/A',
                        'Avg R²': f"{np.mean(r2_values):.3f}" if r2_values else 'N/A',
                        'CV Folds': len(walk_forward)
                    })
                else:
                    comparison_data.append({
                        'Model': model,
                        'Avg MAE': 'N/A',
                        'Avg RMSE': 'N/A',
                        'Avg MAPE': 'N/A',
                        'Avg R²': 'N/A',
                        'CV Folds': 0
                    })

            comparison_df = pd.DataFrame(comparison_data)
            st.dataframe(comparison_df, use_container_width=True)

            # Performance chart - only if we have valid MAE data
            mae_data = []
            model_names = []
            for model in selected_models:
                results = self.results['model_comparison'][model]
                cv_metrics = results.get('cv_metrics', {})
                walk_forward = cv_metrics.get('walk_forward', [])
                
                if walk_forward:
                    mae_values = [r.get('mae', 0) for r in walk_forward if isinstance(r, dict)]
                    if mae_values:
                        mae_data.append(np.mean(mae_values))
                        model_names.append(model)
            
            if mae_data:
                chart_df = pd.DataFrame({
                    'Model': model_names,
                    'MAE': mae_data
                })
                fig = px.bar(
                    chart_df,
                    x='Model',
                    y='MAE',
                    title="Model MAE Comparison",
                    color='Model'
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No valid MAE data available for selected models")

    def show_business_metrics_page(self):
        """
        Show business metrics analysis page.
        """
        st.subheader("Business Metrics Analysis")

        # Business metrics chart
        fig = self.create_business_metrics_chart()
        st.plotly_chart(fig, use_container_width=True)

        # Detailed metrics table
        if self.results and 'model_comparison' in self.results:
            st.subheader("Detailed Performance Metrics")

            metrics_data = []
            for model, results in self.results['model_comparison'].items():
                cv_metrics = results.get('cv_metrics', {})
                walk_forward = cv_metrics.get('walk_forward', [])
                
                if walk_forward:
                    mae_values = [r.get('mae', 0) for r in walk_forward if isinstance(r, dict)]
                    mse_values = [r.get('mse', 0) for r in walk_forward if isinstance(r, dict)]
                    rmse_values = [r.get('rmse', 0) for r in walk_forward if isinstance(r, dict)]
                    mape_values = [r.get('mape', 0) for r in walk_forward if isinstance(r, dict)]
                    r2_values = [r.get('r2', 0) for r in walk_forward if isinstance(r, dict)]
                    
                    metrics_data.append({
                        'Model': model,
                        'Mean MAE': f"{np.mean(mae_values):.2f}" if mae_values else 'N/A',
                        'Std MAE': f"{np.std(mae_values):.2f}" if mae_values else 'N/A',
                        'Mean RMSE': f"{np.mean(rmse_values):.2f}" if rmse_values else 'N/A',
                        'Mean MAPE': f"{np.mean(mape_values):.2f}%" if mape_values else 'N/A',
                        'Mean R²': f"{np.mean(r2_values):.3f}" if r2_values else 'N/A',
                        'Min MAE': f"{np.min(mae_values):.2f}" if mae_values else 'N/A',
                        'Max MAE': f"{np.max(mae_values):.2f}" if mae_values else 'N/A'
                    })
                else:
                    metrics_data.append({
                        'Model': model,
                        'Mean MAE': 'N/A',
                        'Std MAE': 'N/A',
                        'Mean RMSE': 'N/A',
                        'Mean MAPE': 'N/A',
                        'Mean R²': 'N/A',
                        'Min MAE': 'N/A',
                        'Max MAE': 'N/A'
                    })

            metrics_df = pd.DataFrame(metrics_data)
            st.dataframe(metrics_df, use_container_width=True)

    def show_validation_details_page(self):
        """
        Show validation details page.
        """
        st.subheader("Cross-Validation Details")

        if not self.results or 'model_comparison' not in self.results:
            st.error("No validation details available")
            return

        # Model selection for validation details
        models = list(self.results['model_comparison'].keys())
        selected_model = st.selectbox("Select model for validation details", models)

        if selected_model:
            model_results = self.results['model_comparison'][selected_model]
            cv_metrics = model_results.get('cv_metrics', {})

            # Display validation metrics
            st.subheader(f"Validation Metrics for {selected_model}")
            
            walk_forward = cv_metrics.get('walk_forward', [])
            if walk_forward:
                mae_values = [r.get('mae', 0) for r in walk_forward if isinstance(r, dict)]
                mse_values = [r.get('mse', 0) for r in walk_forward if isinstance(r, dict)]
                rmse_values = [r.get('rmse', 0) for r in walk_forward if isinstance(r, dict)]
                mape_values = [r.get('mape', 0) for r in walk_forward if isinstance(r, dict)]
                r2_values = [r.get('r2', 0) for r in walk_forward if isinstance(r, dict)]
                
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    avg_mae = np.mean(mae_values) if mae_values else 0
                    st.metric("Avg MAE", f"{avg_mae:.2f}" if avg_mae > 0 else "N/A")
                with col2:
                    avg_rmse = np.mean(rmse_values) if rmse_values else 0
                    st.metric("Avg RMSE", f"{avg_rmse:.2f}" if avg_rmse > 0 else "N/A")
                with col3:
                    avg_mape = np.mean(mape_values) if mape_values else 0
                    st.metric("Avg MAPE", f"{avg_mape:.2f}%" if avg_mape > 0 else "N/A")
                with col4:
                    avg_r2 = np.mean(r2_values) if r2_values else 0
                    st.metric("Avg R²", f"{avg_r2:.3f}" if len(r2_values) > 0 else "N/A")
                
                # Show validation timeline
                if len(mae_values) > 1:
                    st.subheader("Cross-Validation Performance Over Time")
                    timeline_fig = self.create_validation_timeline_chart(walk_forward)
                    st.plotly_chart(timeline_fig, use_container_width=True)
            else:
                st.warning("No cross-validation metrics available for this model")

            # Statistical tests
            st.subheader("Statistical Tests")
            statistical_tests = model_results.get('statistical_tests', {})
            if statistical_tests:
                test_df = pd.DataFrame({
                    'Test': list(statistical_tests.keys()),
                    'Result': list(statistical_tests.values())
                })
                st.dataframe(test_df, use_container_width=True)
            else:
                st.info("No statistical tests available")

    def show_reports_page(self):
        """
        Show reports and export options page.
        """
        st.subheader("Reports & Export")

        if not self.results:
            st.error("No results available for reporting")
            return

        # Export options
        col1, col2 = st.columns(2)

        with col1:
            if st.button("Export JSON Report"):
                json_str = json.dumps(self.results, indent=2, default=str)
                st.download_button(
                    label="Download JSON",
                    data=json_str,
                    file_name="evaluation_report.json",
                    mime="application/json"
                )

        with col2:
            if st.button("Export CSV Summary"):
                if 'model_comparison' in self.results:
                    summary_data = []
                    for model, results in self.results['model_comparison'].items():
                        cv_metrics = results.get('cv_metrics', {})
                        walk_forward = cv_metrics.get('walk_forward', [])
                        
                        if walk_forward:
                            mae_values = [r.get('mae', 0) for r in walk_forward if isinstance(r, dict)]
                            rmse_values = [r.get('rmse', 0) for r in walk_forward if isinstance(r, dict)]
                            mape_values = [r.get('mape', 0) for r in walk_forward if isinstance(r, dict)]
                            r2_values = [r.get('r2', 0) for r in walk_forward if isinstance(r, dict)]
                            
                            summary_data.append({
                                'model': model,
                                'avg_mae': np.mean(mae_values) if mae_values else 'N/A',
                                'avg_rmse': np.mean(rmse_values) if rmse_values else 'N/A',
                                'avg_mape': np.mean(mape_values) if mape_values else 'N/A',
                                'avg_r2': np.mean(r2_values) if r2_values else 'N/A',
                                'cv_folds': len(walk_forward)
                            })
                        else:
                            summary_data.append({
                                'model': model,
                                'avg_mae': 'N/A',
                                'avg_rmse': 'N/A',
                                'avg_mape': 'N/A',
                                'avg_r2': 'N/A',
                                'cv_folds': 0
                            })

                    summary_df = pd.DataFrame(summary_data)
                    csv = summary_df.to_csv(index=False)
                    st.download_button(
                        label="Download CSV",
                        data=csv,
                        file_name="model_comparison.csv",
                        mime="text/csv"
                    )

        # Raw results display
        with st.expander("Raw Results JSON"):
            st.json(self.results)


def run_evaluation_dashboard(results_path: str = 'reports/model_performance/evaluation_report.json'):
    """
    Run the evaluation dashboard.

    Args:
        results_path (str): Path to evaluation results.
    """
    dashboard = EvaluationDashboard(results_path)
    dashboard.run_dashboard()


if __name__ == "__main__":
    import sys
    results_path = sys.argv[1] if len(sys.argv) > 1 else 'reports/model_performance/evaluation_report.json'
    run_evaluation_dashboard(results_path)

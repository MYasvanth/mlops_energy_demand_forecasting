#!/usr/bin/env python3
"""
Simple Evaluation Dashboard without pandas dependency
"""

import streamlit as st
import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Any

def load_results(results_path: str) -> Dict[str, Any]:
    """Load evaluation results from JSON file."""
    try:
        if Path(results_path).exists():
            with open(results_path, 'r') as f:
                return json.load(f)
        else:
            st.error(f"Results file not found: {results_path}")
            return {}
    except Exception as e:
        st.error(f"Error loading results: {e}")
        return {}

def extract_model_metrics(results: Dict[str, Any]) -> Dict[str, Dict[str, float]]:
    """Extract metrics from model results."""
    model_metrics = {}
    
    if 'model_comparison' not in results:
        return model_metrics
    
    for model_name, model_data in results['model_comparison'].items():
        cv_metrics = model_data.get('cv_metrics', {})
        walk_forward = cv_metrics.get('walk_forward', [])
        
        if walk_forward:
            mae_values = []
            mse_values = []
            rmse_values = []
            mape_values = []
            r2_values = []
            
            for result in walk_forward:
                if isinstance(result, dict):
                    mae_values.append(result.get('mae', 0))
                    mse_values.append(result.get('mse', 0))
                    rmse_values.append(result.get('rmse', 0))
                    mape_values.append(result.get('mape', 0))
                    r2_values.append(result.get('r2', 0))
            
            if mae_values:
                model_metrics[model_name] = {
                    'avg_mae': np.mean(mae_values),
                    'std_mae': np.std(mae_values),
                    'avg_mse': np.mean(mse_values),
                    'avg_rmse': np.mean(rmse_values),
                    'avg_mape': np.mean(mape_values),
                    'avg_r2': np.mean(r2_values),
                    'min_mae': np.min(mae_values),
                    'max_mae': np.max(mae_values),
                    'cv_folds': len(mae_values)
                }
    
    return model_metrics

def show_overview(results: Dict[str, Any], model_metrics: Dict[str, Dict[str, float]]):
    """Show overview page."""
    st.header("Model Performance Overview")
    
    if not model_metrics:
        st.warning("No model metrics available")
        return
    
    # Find best model
    best_model = min(model_metrics.keys(), key=lambda x: model_metrics[x]['avg_mae'])
    best_mae = model_metrics[best_model]['avg_mae']
    
    # Summary metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Models Evaluated", len(model_metrics))
    
    with col2:
        st.metric("Best Model", best_model)
    
    with col3:
        st.metric("Best MAE", f"{best_mae:.2f}")
    
    with col4:
        avg_mae = np.mean([metrics['avg_mae'] for metrics in model_metrics.values()])
        st.metric("Average MAE", f"{avg_mae:.2f}")
    
    # Model comparison table
    st.subheader("Model Comparison")
    
    # Create table data
    table_data = []
    for model_name, metrics in model_metrics.items():
        table_data.append({
            'Model': model_name,
            'Avg MAE': f"{metrics['avg_mae']:.2f}",
            'Std MAE': f"{metrics['std_mae']:.2f}",
            'Avg RMSE': f"{metrics['avg_rmse']:.2f}",
            'Avg MAPE': f"{metrics['avg_mape']:.2f}%",
            'Avg R²': f"{metrics['avg_r2']:.3f}",
            'CV Folds': metrics['cv_folds']
        })
    
    # Display as table
    for i, row in enumerate(table_data):
        if i == 0:
            # Header
            cols = st.columns(len(row))
            for j, (key, value) in enumerate(row.items()):
                cols[j].write(f"**{key}**")
        
        cols = st.columns(len(row))
        for j, (key, value) in enumerate(row.items()):
            cols[j].write(str(value))

def show_detailed_comparison(model_metrics: Dict[str, Dict[str, float]]):
    """Show detailed model comparison."""
    st.header("Detailed Model Comparison")
    
    if not model_metrics:
        st.warning("No model metrics available")
        return
    
    # Model selection
    selected_models = st.multiselect(
        "Select models to compare",
        list(model_metrics.keys()),
        default=list(model_metrics.keys())[:3]  # Select first 3 by default
    )
    
    if not selected_models:
        st.info("Please select at least one model")
        return
    
    # Comparison table
    st.subheader("Performance Metrics")
    
    for model in selected_models:
        metrics = model_metrics[model]
        
        st.write(f"**{model}**")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("MAE", f"{metrics['avg_mae']:.2f} ± {metrics['std_mae']:.2f}")
        
        with col2:
            st.metric("RMSE", f"{metrics['avg_rmse']:.2f}")
        
        with col3:
            st.metric("MAPE", f"{metrics['avg_mape']:.2f}%")
        
        with col4:
            st.metric("R²", f"{metrics['avg_r2']:.3f}")
        
        # Additional details
        with st.expander(f"Details for {model}"):
            st.write(f"- **Cross-validation folds:** {metrics['cv_folds']}")
            st.write(f"- **Min MAE:** {metrics['min_mae']:.2f}")
            st.write(f"- **Max MAE:** {metrics['max_mae']:.2f}")
            st.write(f"- **MAE Range:** {metrics['max_mae'] - metrics['min_mae']:.2f}")

def show_validation_details(results: Dict[str, Any], model_metrics: Dict[str, Dict[str, float]]):
    """Show validation details."""
    st.header("Cross-Validation Details")
    
    if not model_metrics:
        st.warning("No model metrics available")
        return
    
    # Model selection
    selected_model = st.selectbox("Select model", list(model_metrics.keys()))
    
    if not selected_model:
        return
    
    # Show metrics for selected model
    metrics = model_metrics[selected_model]
    
    st.subheader(f"Validation Metrics for {selected_model}")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Average MAE", f"{metrics['avg_mae']:.2f}")
    
    with col2:
        st.metric("Standard Deviation", f"{metrics['std_mae']:.2f}")
    
    with col3:
        st.metric("Min MAE", f"{metrics['min_mae']:.2f}")
    
    with col4:
        st.metric("Max MAE", f"{metrics['max_mae']:.2f}")
    
    # Show raw CV results
    model_data = results['model_comparison'][selected_model]
    cv_metrics = model_data.get('cv_metrics', {})
    walk_forward = cv_metrics.get('walk_forward', [])
    
    if walk_forward:
        st.subheader("Cross-Validation Results")
        
        # Show first few results
        st.write("**Sample CV Results:**")
        for i, result in enumerate(walk_forward[:5]):  # Show first 5
            if isinstance(result, dict):
                st.write(f"Fold {i+1}: MAE={result.get('mae', 'N/A'):.2f}, "
                        f"RMSE={result.get('rmse', 'N/A'):.2f}, "
                        f"MAPE={result.get('mape', 'N/A'):.2f}%, "
                        f"R²={result.get('r2', 'N/A'):.3f}")
        
        if len(walk_forward) > 5:
            st.write(f"... and {len(walk_forward) - 5} more folds")

def main():
    """Main dashboard function."""
    st.set_page_config(
        page_title="Energy Demand Forecasting - Model Evaluation",
        page_icon="⚡",
        layout="wide"
    )
    
    st.title("⚡ Energy Demand Forecasting - Model Evaluation Dashboard")
    st.markdown("---")
    
    # Load results
    results_path = st.sidebar.text_input(
        "Results file path", 
        value="reports/model_performance/evaluation_report.json"
    )
    
    if st.sidebar.button("Load Results"):
        st.rerun()
    
    results = load_results(results_path)
    
    if not results:
        st.error("No results loaded. Please check the file path.")
        return
    
    # Extract metrics
    model_metrics = extract_model_metrics(results)
    
    # Sidebar navigation
    st.sidebar.header("Navigation")
    page = st.sidebar.radio(
        "Select View",
        ["Overview", "Model Comparison", "Validation Details", "Raw Data"]
    )
    
    # Show selected page
    if page == "Overview":
        show_overview(results, model_metrics)
    elif page == "Model Comparison":
        show_detailed_comparison(model_metrics)
    elif page == "Validation Details":
        show_validation_details(results, model_metrics)
    elif page == "Raw Data":
        st.header("Raw Results")
        st.json(results)

if __name__ == "__main__":
    main()
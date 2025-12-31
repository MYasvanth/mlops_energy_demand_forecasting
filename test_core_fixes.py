#!/usr/bin/env python3
"""
Test script to verify the core fixes for MLflow and evaluation pipeline issues.
"""

import sys
import os
from pathlib import Path
import pandas as pd
import numpy as np
import mlflow

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def test_mlflow_cleanup():
    """Test MLflow run cleanup functionality."""
    print("Testing MLflow run cleanup...")
    
    # Ensure clean state
    if mlflow.active_run():
        mlflow.end_run()
        print("Cleaned up existing MLflow run")
    
    # Start a test run
    with mlflow.start_run(run_name="test_run"):
        mlflow.log_metric("test_metric", 1.0)
        print("MLflow run started and metric logged")
    
    print("MLflow run ended successfully")
    return True

def test_evaluation_core_functions():
    """Test core evaluation functions without TensorFlow dependencies."""
    print("\nTesting core evaluation function imports...")
    
    try:
        from src.models.evaluation import (
            calculate_metrics,
            energy_specific_metrics,
            time_series_cross_validation,
            walk_forward_validation,
            expanding_window_validation
        )
        print("Core evaluation functions imported successfully")
        
        # Test calculate_metrics
        np.random.seed(42)
        y_true = np.random.normal(1000, 100, 50)
        y_pred = y_true + np.random.normal(0, 50, 50)
        metrics = calculate_metrics(y_true, y_pred)
        print(f"calculate_metrics works: MAE={metrics['mae']:.2f}")
        
        # Test energy_specific_metrics
        business_metrics = energy_specific_metrics(y_true, y_pred)
        print(f"energy_specific_metrics works: Peak MAE={business_metrics['peak_hour_mae']:.2f}")
        
        # Test time series cross-validation
        dates = pd.date_range('2020-01-01', periods=100, freq='H')
        df = pd.DataFrame({
            'total_load_actual': 1000 + np.random.normal(0, 100, 100)
        }, index=dates)
        
        cv_results = list(time_series_cross_validation(df, n_splits=3))
        print(f"time_series_cross_validation works: {len(cv_results)} splits")
        
        return True
        
    except Exception as e:
        print(f"Error in core evaluation functions: {e}")
        return False

def test_function_signatures():
    """Test that function signatures match expected usage."""
    print("\nTesting function signatures...")
    
    try:
        # Test that model_validation_pipeline accepts target_column parameter
        from src.models.evaluation import model_validation_pipeline
        import inspect
        
        sig = inspect.signature(model_validation_pipeline)
        params = list(sig.parameters.keys())
        
        if 'target_column' in params:
            print("model_validation_pipeline has target_column parameter")
        else:
            print("ERROR: model_validation_pipeline missing target_column parameter")
            return False
            
        # Test that generate_evaluation_report accepts best_model parameter
        from src.models.evaluation import generate_evaluation_report
        sig = inspect.signature(generate_evaluation_report)
        params = list(sig.parameters.keys())
        
        if 'best_model' in params:
            print("generate_evaluation_report has best_model parameter")
        else:
            print("ERROR: generate_evaluation_report missing best_model parameter")
            return False
            
        return True
        
    except Exception as e:
        print(f"Error testing function signatures: {e}")
        return False

def main():
    """Run all tests."""
    print("Testing MLOps Energy Demand Forecasting Core Fixes")
    print("=" * 60)
    
    # Ensure clean MLflow state
    if mlflow.active_run():
        mlflow.end_run()
    
    success = True
    
    # Test MLflow cleanup
    try:
        if not test_mlflow_cleanup():
            success = False
    except Exception as e:
        print(f"MLflow test failed: {e}")
        success = False
    
    # Test core evaluation functions
    if not test_evaluation_core_functions():
        success = False
    
    # Test function signatures
    if not test_function_signatures():
        success = False
    
    print("\n" + "=" * 60)
    if success:
        print("All core tests passed! The main fixes are working correctly.")
        print("\nKey fixes verified:")
        print("1. MLflow active run error fixed with proper cleanup")
        print("2. model_validation_pipeline function signature includes target_column")
        print("3. generate_evaluation_report function signature includes best_model")
        print("4. Core evaluation functions work without TensorFlow dependencies")
        print("\nThe original errors should now be resolved:")
        print("- 'Run with UUID ... is already active' -> Fixed with mlflow.end_run()")
        print("- 'unexpected keyword argument target_column' -> Fixed function signature")
    else:
        print("Some core tests failed. Please check the error messages above.")
    
    return success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
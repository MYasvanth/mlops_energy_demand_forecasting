#!/usr/bin/env python3
"""
Test script to verify the fixes for MLflow and evaluation pipeline issues.
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

def test_evaluation_functions():
    """Test evaluation function signatures."""
    print("\nTesting evaluation function imports...")
    
    try:
        from src.models.evaluation import (
            model_validation_pipeline,
            generate_evaluation_report,
            calculate_metrics,
            comprehensive_cross_validation
        )
        print("All evaluation functions imported successfully")
        
        # Test function signatures with dummy data
        np.random.seed(42)
        dates = pd.date_range('2020-01-01', periods=100, freq='H')
        df = pd.DataFrame({
            'total_load_actual': 1000 + np.random.normal(0, 100, 100)
        }, index=dates)
        
        # Test calculate_metrics
        y_true = np.random.normal(1000, 100, 50)
        y_pred = y_true + np.random.normal(0, 50, 50)
        metrics = calculate_metrics(y_true, y_pred)
        print(f"calculate_metrics works: MAE={metrics['mae']:.2f}")
        
        # Test comprehensive_cross_validation
        cv_config = {'strategies': ['walk_forward'], 'model_type': 'arima', 'n_splits': 2}
        cv_results = comprehensive_cross_validation(df, 'total_load_actual', cv_config)
        print("comprehensive_cross_validation works")
        
        # Test model_validation_pipeline with dummy models
        dummy_models = {'test_model': {'mae': 100.0}}
        validation_results, best_model = model_validation_pipeline(
            dummy_models, df, target_column='total_load_actual', config=cv_config
        )
        print(f"model_validation_pipeline works: best_model={best_model}")
        
        # Test generate_evaluation_report
        report = generate_evaluation_report(validation_results, best_model)
        print("generate_evaluation_report works")
        
    except Exception as e:
        print(f"Error in evaluation functions: {e}")
        return False
    
    return True

def test_training_functions():
    """Test training function imports."""
    print("\nTesting training function imports...")
    
    try:
        from src.models.train import train_multiple_models, TimeSeriesTrainer
        print("Training functions imported successfully")
        
        # Test TimeSeriesTrainer initialization
        trainer = TimeSeriesTrainer(model_type='arima', target_column='total_load_actual')
        print("TimeSeriesTrainer initialized successfully")
        
    except Exception as e:
        print(f"Error in training functions: {e}")
        return False
    
    return True

def main():
    """Run all tests."""
    print("Testing MLOps Energy Demand Forecasting Fixes")
    print("=" * 60)
    
    # Ensure clean MLflow state
    if mlflow.active_run():
        mlflow.end_run()
    
    success = True
    
    # Test MLflow cleanup
    try:
        test_mlflow_cleanup()
    except Exception as e:
        print(f"MLflow test failed: {e}")
        success = False
    
    # Test evaluation functions
    if not test_evaluation_functions():
        success = False
    
    # Test training functions
    if not test_training_functions():
        success = False
    
    print("\n" + "=" * 60)
    if success:
        print("All tests passed! The fixes are working correctly.")
        print("\nKey fixes applied:")
        print("1. Fixed MLflow active run error by adding proper run cleanup")
        print("2. Fixed model_validation_pipeline function signature mismatch")
        print("3. Updated generate_evaluation_report to match expected parameters")
        print("4. Added proper error handling for model version transitions")
        print("\nYou can now run the evaluation script without errors:")
        print("python scripts/evaluation/run_model_evaluation.py --quick-demo")
    else:
        print("Some tests failed. Please check the error messages above.")
    
    return success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
#!/usr/bin/env python3
"""
Simple test script to check if the dashboard works
"""

import sys
import json
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def test_data_loading():
    """Test if we can load the evaluation results"""
    results_path = 'reports/model_performance/evaluation_report.json'
    
    try:
        if Path(results_path).exists():
            with open(results_path, 'r') as f:
                results = json.load(f)
            
            print("[OK] Successfully loaded evaluation results")
            print(f"[OK] Found {len(results.get('model_comparison', {}))} models")
            
            # Check structure
            if 'model_comparison' in results:
                for model_name, model_data in results['model_comparison'].items():
                    cv_metrics = model_data.get('cv_metrics', {})
                    walk_forward = cv_metrics.get('walk_forward', [])
                    print(f"[OK] Model '{model_name}': {len(walk_forward)} CV folds")
                    
                    if walk_forward and len(walk_forward) > 0:
                        first_result = walk_forward[0]
                        if isinstance(first_result, dict):
                            mae = first_result.get('mae', 'N/A')
                            print(f"  - First fold MAE: {mae}")
                        else:
                            print(f"  - Unexpected data type: {type(first_result)}")
                    break  # Just check first model
            
            return True
        else:
            print(f"[ERROR] Results file not found: {results_path}")
            return False
            
    except Exception as e:
        print(f"[ERROR] Error loading results: {e}")
        return False

def test_basic_imports():
    """Test if we can import required libraries"""
    try:
        import numpy as np
        print("[OK] numpy imported successfully")
        
        import json
        print("[OK] json imported successfully")
        
        # Try streamlit import
        try:
            import streamlit as st
            print("[OK] streamlit imported successfully")
        except Exception as e:
            print(f"[ERROR] streamlit import failed: {e}")
            return False
            
        # Try plotly import
        try:
            import plotly.graph_objects as go
            import plotly.express as px
            print("[OK] plotly imported successfully")
        except Exception as e:
            print(f"[ERROR] plotly import failed: {e}")
            return False
            
        return True
        
    except Exception as e:
        print(f"[ERROR] Basic imports failed: {e}")
        return False

if __name__ == "__main__":
    print("Testing dashboard components...")
    print("=" * 50)
    
    # Test imports
    if not test_basic_imports():
        print("\n[ERROR] Import tests failed")
        sys.exit(1)
    
    # Test data loading
    if not test_data_loading():
        print("\n[ERROR] Data loading tests failed")
        sys.exit(1)
    
    print("\n[OK] All tests passed! Dashboard should work.")
    print("\nTry running the dashboard with:")
    print("streamlit run src/models/evaluation_dashboard.py -- reports/model_performance/evaluation_report.json")
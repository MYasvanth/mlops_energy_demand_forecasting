#!/usr/bin/env python3
"""
Diagnose FastAPI app issues
"""
import sys
sys.path.append('.')

def diagnose_imports():
    """Diagnose import issues step by step"""
    
    print("=== FastAPI Import Diagnosis ===")
    
    # Step 1: Basic imports
    try:
        import pandas as pd
        print("[OK] Pandas imported")
    except Exception as e:
        print(f"[FAIL] Pandas: {e}")
    
    try:
        import numpy as np
        print("[OK] Numpy imported")
    except Exception as e:
        print(f"[FAIL] Numpy: {e}")
    
    try:
        from fastapi import FastAPI
        print("[OK] FastAPI imported")
    except Exception as e:
        print(f"[FAIL] FastAPI: {e}")
    
    # Step 2: Project imports
    try:
        from src.models.predict import TimeSeriesPredictor
        print("[OK] TimeSeriesPredictor imported")
    except Exception as e:
        print(f"[FAIL] TimeSeriesPredictor: {e}")
    
    try:
        from src.monitoring.evidently_monitoring import EvidentlyMonitor
        print("[OK] EvidentlyMonitor imported")
    except Exception as e:
        print(f"[FAIL] EvidentlyMonitor: {e}")
    
    # Step 3: FastAPI app
    try:
        from src.deployment.fastapi_app import app
        print(f"[OK] FastAPI app imported: {app.title}")
        return app
    except Exception as e:
        print(f"[FAIL] FastAPI app: {e}")
        return None

def test_app_functionality(app):
    """Test app functionality if available"""
    
    if app is None:
        print("\n[SKIP] App not available for testing")
        return
    
    print(f"\n=== Testing App: {app.title} ===")
    
    try:
        from fastapi.testclient import TestClient
        client = TestClient(app)
        
        # Test available routes
        routes = [route.path for route in app.routes if hasattr(route, 'path')]
        print(f"Available routes: {routes}")
        
        # Test basic endpoints
        basic_endpoints = ["/", "/health", "/models", "/monitoring/status"]
        
        for endpoint in basic_endpoints:
            if endpoint in [route.path for route in app.routes if hasattr(route, 'path')]:
                try:
                    response = client.get(endpoint)
                    print(f"[OK] {endpoint}: {response.status_code}")
                    if response.status_code == 200:
                        data = response.json()
                        if isinstance(data, dict) and len(str(data)) < 200:
                            print(f"    {data}")
                except Exception as e:
                    print(f"[FAIL] {endpoint}: {e}")
            else:
                print(f"[SKIP] {endpoint}: Route not found")
                
    except Exception as e:
        print(f"[FAIL] Testing failed: {e}")

def check_model_files():
    """Check if model files exist"""
    
    print("\n=== Model Files Check ===")
    
    import os
    from pathlib import Path
    
    model_paths = [
        "models/arima_model_fitted.pkl",
        "models/prophet_model_prophet.pkl", 
        "models/lstm_model_lstm.h5",
        "models/production/",
        "models/staging/"
    ]
    
    for path in model_paths:
        if os.path.exists(path):
            print(f"[OK] {path} exists")
        else:
            print(f"[MISSING] {path}")

if __name__ == "__main__":
    app = diagnose_imports()
    test_app_functionality(app)
    check_model_files()
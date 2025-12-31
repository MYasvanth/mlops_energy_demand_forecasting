#!/usr/bin/env python3
"""
Test dashboard dependencies
"""

def test_dashboard_dependencies():
    """Test if dashboard dependencies are available"""
    
    print("=== Dashboard Dependencies Test ===")
    
    missing = []
    available = []
    
    # Test Streamlit
    try:
        import streamlit as st
        available.append("streamlit")
        print("[OK] Streamlit available")
    except ImportError:
        missing.append("streamlit")
        print("[MISSING] Streamlit not installed")
    
    # Test Plotly
    try:
        import plotly.graph_objects as go
        import plotly.express as px
        available.append("plotly")
        print("[OK] Plotly available")
    except ImportError:
        missing.append("plotly")
        print("[MISSING] Plotly not installed")
    
    # Test Pandas (might have numpy issue)
    try:
        import pandas as pd
        available.append("pandas")
        print("[OK] Pandas available")
    except ImportError as e:
        missing.append("pandas")
        print(f"[MISSING] Pandas not available: {e}")
    
    # Test JSON
    try:
        import json
        available.append("json")
        print("[OK] JSON available")
    except ImportError:
        missing.append("json")
        print("[MISSING] JSON not available")
    
    # Check evaluation report
    import os
    report_exists = os.path.exists("reports/model_performance/evaluation_report.json")
    print(f"[{'OK' if report_exists else 'MISSING'}] Evaluation report: {report_exists}")
    
    # Summary
    print(f"\n=== Summary ===")
    print(f"Available: {', '.join(available) if available else 'None'}")
    print(f"Missing: {', '.join(missing) if missing else 'None'}")
    
    if missing:
        print(f"\nTo install missing packages:")
        print(f"pip install {' '.join(missing)}")
        return False
    else:
        print("\nAll dependencies available!")
        return True

if __name__ == "__main__":
    test_dashboard_dependencies()
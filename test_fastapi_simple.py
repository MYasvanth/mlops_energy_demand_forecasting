#!/usr/bin/env python3
"""
Simple FastAPI test script
"""
import sys
import os
sys.path.append('.')

def test_fastapi_endpoints():
    """Test FastAPI endpoints functionality"""
    
    print("=== FastAPI Endpoint Analysis ===")
    
    # Test 1: Import check
    try:
        from fastapi import FastAPI
        from fastapi.testclient import TestClient
        print("[OK] FastAPI imports successful")
    except ImportError as e:
        print(f"[FAIL] FastAPI import failed: {e}")
        return False
    
    # Test 2: App import
    try:
        from src.deployment.fastapi_app import app
        print(f"[OK] App imported: {app.title}")
    except Exception as e:
        print(f"[FAIL] App import failed: {e}")
        return False
    
    # Test 3: Create test client
    try:
        client = TestClient(app)
        print("[OK] Test client created")
    except Exception as e:
        print(f"[FAIL] Test client failed: {e}")
        return False
    
    # Test 4: Test endpoints
    endpoints_status = {}
    
    # Root endpoint
    try:
        response = client.get("/")
        endpoints_status["root"] = {
            "status": response.status_code,
            "working": response.status_code == 200,
            "response": response.json() if response.status_code == 200 else None
        }
    except Exception as e:
        endpoints_status["root"] = {"status": "error", "working": False, "error": str(e)}
    
    # Health endpoint
    try:
        response = client.get("/health")
        endpoints_status["health"] = {
            "status": response.status_code,
            "working": response.status_code == 200,
            "response": response.json() if response.status_code == 200 else None
        }
    except Exception as e:
        endpoints_status["health"] = {"status": "error", "working": False, "error": str(e)}
    
    # Models endpoint
    try:
        response = client.get("/models")
        endpoints_status["models"] = {
            "status": response.status_code,
            "working": response.status_code == 200,
            "response": response.json() if response.status_code == 200 else None
        }
    except Exception as e:
        endpoints_status["models"] = {"status": "error", "working": False, "error": str(e)}
    
    # Monitoring status
    try:
        response = client.get("/monitoring/status")
        endpoints_status["monitoring"] = {
            "status": response.status_code,
            "working": response.status_code == 200,
            "response": response.json() if response.status_code == 200 else None
        }
    except Exception as e:
        endpoints_status["monitoring"] = {"status": "error", "working": False, "error": str(e)}
    
    # Print results
    print("\n=== Endpoint Test Results ===")
    for endpoint, result in endpoints_status.items():
        status_icon = "[OK]" if result["working"] else "[FAIL]"
        print(f"{status_icon} /{endpoint}: {result['status']}")
        if result["working"] and result.get("response"):
            print(f"   Response: {result['response']}")
        elif not result["working"] and result.get("error"):
            print(f"   Error: {result['error']}")
    
    # Summary
    working_count = sum(1 for r in endpoints_status.values() if r["working"])
    total_count = len(endpoints_status)
    
    print(f"\n=== Summary ===")
    print(f"Working endpoints: {working_count}/{total_count}")
    print(f"Success rate: {working_count/total_count*100:.1f}%")
    
    return working_count == total_count

if __name__ == "__main__":
    test_fastapi_endpoints()
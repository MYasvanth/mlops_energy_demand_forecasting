#!/usr/bin/env python3
"""
Basic FastAPI functionality test
"""

def test_fastapi_basic():
    """Test basic FastAPI functionality without problematic imports"""
    
    print("=== FastAPI Basic Test ===")
    
    # Test 1: Core FastAPI
    try:
        from fastapi import FastAPI, HTTPException
        from pydantic import BaseModel
        print("[OK] FastAPI core imports successful")
    except Exception as e:
        print(f"[FAIL] FastAPI core imports: {e}")
        return False
    
    # Test 2: Create simple app
    try:
        app = FastAPI(title="Test Energy API", version="1.0.0")
        
        class HealthResponse(BaseModel):
            status: str
            message: str
        
        @app.get("/", response_model=dict)
        def root():
            return {"message": "Energy Demand Forecasting API", "status": "running"}
        
        @app.get("/health", response_model=HealthResponse)
        def health():
            return HealthResponse(status="healthy", message="API is running")
        
        @app.get("/models")
        def models():
            return {"models": [], "count": 0}
        
        print("[OK] Simple FastAPI app created")
    except Exception as e:
        print(f"[FAIL] App creation: {e}")
        return False
    
    # Test 3: Test client
    try:
        from fastapi.testclient import TestClient
        client = TestClient(app)
        print("[OK] Test client created")
    except Exception as e:
        print(f"[FAIL] Test client: {e}")
        return False
    
    # Test 4: Test endpoints
    print("\n=== Testing Endpoints ===")
    
    endpoints = ["/", "/health", "/models"]
    results = {}
    
    for endpoint in endpoints:
        try:
            response = client.get(endpoint)
            results[endpoint] = {
                "status_code": response.status_code,
                "working": response.status_code == 200,
                "response": response.json() if response.status_code == 200 else None
            }
            status = "[OK]" if response.status_code == 200 else "[FAIL]"
            print(f"{status} {endpoint}: {response.status_code}")
            if response.status_code == 200:
                print(f"    Response: {response.json()}")
        except Exception as e:
            results[endpoint] = {"status_code": "error", "working": False, "error": str(e)}
            print(f"[FAIL] {endpoint}: {e}")
    
    # Summary
    working = sum(1 for r in results.values() if r.get("working", False))
    total = len(results)
    
    print(f"\n=== Summary ===")
    print(f"Working endpoints: {working}/{total}")
    print(f"Success rate: {working/total*100:.1f}%")
    
    return working == total

if __name__ == "__main__":
    success = test_fastapi_basic()
    print(f"\nOverall test result: {'PASS' if success else 'FAIL'}")
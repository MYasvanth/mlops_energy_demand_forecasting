#!/usr/bin/env python3
"""
Test FastAPI endpoints
"""
import requests
import json

# Test health endpoint
def test_health():
    response = requests.get("http://localhost:8000/health")
    print("Health Check:", response.json())

# Test models endpoint  
def test_models():
    response = requests.get("http://localhost:8000/models")
    print("Available Models:", response.json())

# Test prediction
def test_prediction():
    data = {
        "model_name": "prophet",
        "hours_ahead": 24,
        "data_path": "data/processed/processed_energy_weather.csv"
    }
    response = requests.post("http://localhost:8000/predict", json=data)
    print("Prediction Response:", response.json())

if __name__ == "__main__":
    test_health()
    test_models() 
    test_prediction()
#!/usr/bin/env python3
"""Test the refactored evaluation system."""

import sys
import pandas as pd
import numpy as np
from pathlib import Path
import json

# Add src to path
sys.path.append('src')

from models.core.validation import evaluate_all_models

def create_sample_data():
    """Create sample time series data."""
    dates = pd.date_range('2020-01-01', periods=200, freq='H')
    np.random.seed(42)
    
    # Create realistic energy demand pattern
    hourly_pattern = np.sin(np.arange(200) * 2 * np.pi / 24)
    daily_pattern = np.sin(np.arange(200) * 2 * np.pi / (24 * 7))
    noise = np.random.normal(0, 50, 200)
    
    data = pd.DataFrame({
        'total_load_actual': 1000 + 200 * hourly_pattern + 100 * daily_pattern + noise
    }, index=dates)
    
    return data

def main():
    """Run refactored evaluation test."""
    print("🚀 Testing Refactored Evaluation System")
    print("=" * 50)
    
    # Create sample data
    print("📊 Creating sample data...")
    data = create_sample_data()
    print(f"Data shape: {data.shape}")
    
    # Run evaluation
    print("\n🔄 Running evaluation...")
    results = evaluate_all_models(
        data=data,
        target_column='total_load_actual',
        models=['baseline', 'arima']  # Start with fast models
    )
    
    # Display results
    print("\n📈 Results:")
    print("-" * 30)
    
    for model, result in results['model_comparison'].items():
        mae = result['mae']
        cv_results = result['cv_metrics']['walk_forward']
        print(f"{model.upper()}: MAE={mae:.2f}, CV Folds={len(cv_results)}")
    
    print(f"\n🏆 Best Model: {results['best_model']}")
    
    # Save results
    output_path = Path('reports/model_performance/evaluation_report.json')
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"💾 Results saved to: {output_path}")
    print("\n✅ Refactored evaluation completed successfully!")

if __name__ == "__main__":
    main()
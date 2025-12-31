#!/usr/bin/env python3
"""
Run Model Evaluation Script

This script runs comprehensive model evaluation with cross-validation
and generates evaluation reports and dashboards.
"""

import sys
import argparse
import json
from pathlib import Path
from typing import Dict, Any

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.models.evaluation import (
    model_validation_pipeline,
    generate_evaluation_report,
    comprehensive_cross_validation,
    calculate_metrics,
    energy_specific_metrics
)
from src.models.train import train_multiple_models
from src.data.preprocessing import full_preprocessing_pipeline
from src.data.ingestion import ingest_data
from src.features.feature_engineering import full_feature_engineering_pipeline


def load_sample_data() -> Dict[str, Any]:
    """
    Load sample data for evaluation.

    Returns:
        Dict[str, Any]: Processed data dictionary.
    """
    try:
        # Load sample data
        energy_path = project_root / 'data' / 'raw' / 'energy_dataset.csv'
        weather_path = project_root / 'data' / 'raw' / 'weather_features.csv'

        if not energy_path.exists() or not weather_path.exists():
            print("Warning: Sample data files not found. Using synthetic data for demonstration.")

            # Generate synthetic data
            import pandas as pd
            import numpy as np

            dates = pd.date_range('2020-01-01', periods=1000, freq='H')
            np.random.seed(42)

            data = pd.DataFrame({
                'time': dates,
                'total_load_actual': 1000 + np.random.normal(0, 100, 1000),
                'total_load_forecast': 1000 + np.random.normal(0, 110, 1000),
                'price_day_ahead': 50 + np.random.normal(0, 10, 1000)
            })

            return {'processed_data': data, 'target_column': 'total_load_actual'}

        # Load real data
        raw_data = ingest_data(str(energy_path), str(weather_path))
        processed_data = full_preprocessing_pipeline(raw_data['energy'], raw_data['weather'])
        feature_data = full_feature_engineering_pipeline(processed_data)

        return {
            'processed_data': feature_data,
            'target_column': 'total_load_actual'
        }

    except Exception as e:
        print(f"Error loading data: {e}")
        print("Using synthetic data for demonstration.")

        # Fallback to synthetic data
        import pandas as pd
        import numpy as np

        dates = pd.date_range('2020-01-01', periods=1000, freq='H')
        np.random.seed(42)

        data = pd.DataFrame({
            'time': dates,
            'total_load_actual': 1000 + np.random.normal(0, 100, 1000),
            'total_load_forecast': 1000 + np.random.normal(0, 110, 1000),
            'price_day_ahead': 50 + np.random.normal(0, 10, 1000)
        })

        return {'processed_data': data, 'target_column': 'total_load_actual'}


def run_evaluation_demo(data: Dict[str, Any], models_to_evaluate: list = None,
                       enable_dashboard: bool = True) -> Dict[str, Any]:
    """
    Run comprehensive model evaluation demo.

    Args:
        data (Dict[str, Any]): Processed data dictionary.
        models_to_evaluate (list): List of models to evaluate.
        enable_dashboard (bool): Whether to enable dashboard generation.

    Returns:
        Dict[str, Any]: Evaluation results.
    """
    if models_to_evaluate is None:
        models_to_evaluate = ['arima', 'prophet', 'lstm']

    df = data['processed_data']
    target_column = data['target_column']

    print(f"Starting evaluation of {len(models_to_evaluate)} models...")
    print(f"Data shape: {df.shape}")
    print(f"Target column: {target_column}")
    print("-" * 50)

    # Train models
    print("Training models...")
    trained_models = train_multiple_models(
        df,
        target_column=target_column,
        models=models_to_evaluate
    )

    # Extract model results for validation
    models_for_validation = {}
    for model_name in models_to_evaluate:
        if model_name in trained_models:
            models_for_validation[model_name] = trained_models[model_name]

    # Run comprehensive validation
    print("Running cross-validation...")
    validation_config = {
        'strategies': ['walk_forward', 'expanding_window'],
        'model_type': 'arima',  # Default, will be overridden per model
        'n_splits': 3
    }

    validation_results, best_model = model_validation_pipeline(
        models_for_validation,
        df,
        target_column=target_column,
        config=validation_config
    )

    # Generate evaluation report
    print("Generating evaluation report...")
    report = generate_evaluation_report(validation_results, best_model)

    # Add training results to report
    report['training_results'] = trained_models

    print("\nEvaluation completed!")
    print(f"Best model: {best_model}")
    print(f"Report saved to: reports/model_performance/evaluation_report.json")

    if enable_dashboard:
        print("\nTo view the interactive dashboard, run:")
        print("python scripts/evaluation/run_evaluation_dashboard.py")

    return {
        'validation_results': validation_results,
        'best_model': best_model,
        'report': report,
        'training_results': trained_models
    }


def main():
    """Main entry point for model evaluation."""
    parser = argparse.ArgumentParser(
        description="Run comprehensive model evaluation",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        '--models',
        nargs='+',
        default=['arima', 'prophet', 'lstm'],
        choices=['arima', 'prophet', 'lstm'],
        help='Models to evaluate'
    )

    parser.add_argument(
        '--output-path',
        type=str,
        default='reports/model_performance/evaluation_report.json',
        help='Path to save evaluation report'
    )

    parser.add_argument(
        '--no-dashboard',
        action='store_true',
        help='Disable dashboard generation'
    )

    parser.add_argument(
        '--quick-demo',
        action='store_true',
        help='Run quick demo with synthetic data'
    )

    args = parser.parse_args()

    print("Energy Demand Forecasting - Model Evaluation")
    print("=" * 60)

    try:
        # Ensure clean MLflow state
        import mlflow
        if mlflow.active_run():
            mlflow.end_run()
        # Load data
        if args.quick_demo:
            print("Running quick demo with synthetic data...")
            data = load_sample_data()
        else:
            print("Loading and processing data...")
            data = load_sample_data()

        # Run evaluation
        results = run_evaluation_demo(
            data,
            models_to_evaluate=args.models,
            enable_dashboard=not args.no_dashboard
        )

        # Save detailed results
        output_path = Path(args.output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w') as f:
            json.dump(results['report'], f, indent=2, default=str)

        print(f"\n✅ Evaluation completed successfully!")
        print(f"📊 Results saved to: {output_path}")
        print(f"🏆 Best model: {results['best_model']}")

        # Print summary
        if 'summary' in results['report']:
            summary = results['report']['summary']
            print("\n📈 Summary:")
            print(f"   Models evaluated: {summary.get('total_models_evaluated', 'N/A')}")
            print(f"   Best MAE: {summary.get('best_model_mae', 'N/A')}")
            print(f"   Average MAE: {summary.get('average_mae', 'N/A')}")

    except Exception as e:
        print(f"❌ Error during evaluation: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

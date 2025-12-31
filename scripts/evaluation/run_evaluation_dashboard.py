#!/usr/bin/env python3
"""
Run Evaluation Dashboard Script

This script launches the interactive evaluation dashboard for model comparison
and cross-validation results visualization.
"""

import sys
import argparse
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.models.evaluation_dashboard import run_evaluation_dashboard


def main():
    """Main entry point for the evaluation dashboard."""
    parser = argparse.ArgumentParser(
        description="Run interactive evaluation dashboard",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        '--results-path',
        type=str,
        default='reports/model_performance/evaluation_report.json',
        help='Path to evaluation results JSON file'
    )

    parser.add_argument(
        '--port',
        type=int,
        default=8501,
        help='Port to run the dashboard on'
    )

    args = parser.parse_args()

    # Check if results file exists
    results_path = Path(args.results_path)
    if not results_path.exists():
        print(f"Warning: Results file not found at {results_path}")
        print("The dashboard will show empty results. Run model evaluation first.")

    print(f"Starting evaluation dashboard on port {args.port}...")
    print(f"Results path: {args.results_path}")
    print(f"Open your browser to http://localhost:{args.port}")
    print("-" * 50)

    # Import streamlit here to avoid import errors if not installed
    try:
        import subprocess
        import sys
        
        # Run streamlit with proper port configuration
        cmd = [
            sys.executable, '-m', 'streamlit', 'run',
            'src/models/evaluation_dashboard.py',
            '--server.port', str(args.port),
            '--server.address', 'localhost',
            '--', str(results_path)
        ]
        
        print(f"Running: {' '.join(cmd)}")
        subprocess.run(cmd)

    except ImportError:
        print("Error: Streamlit is not installed.")
        print("Install it with: pip install streamlit")
        sys.exit(1)
    except Exception as e:
        print(f"Error running dashboard: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()

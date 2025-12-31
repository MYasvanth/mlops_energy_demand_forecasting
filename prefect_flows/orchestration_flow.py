"""
Prefect Orchestration Flow for Energy Demand Forecasting

This module defines a Prefect flow for orchestrating the complete ML pipeline,
including scheduling, retries, monitoring, and integration with ZenML.
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, Any, Optional
from pathlib import Path

import pandas as pd
from prefect import flow, task
from prefect.states import Completed, Failed
from prefect.logging import get_run_logger
from prefect.server.schemas.core import FlowRun
from prefect.server.schemas.states import StateType


@task(retries=3, retry_delay_seconds=60)
def validate_data_availability(energy_path: str, weather_path: str) -> bool:
    """
    Validate that required data files are available.

    Args:
        energy_path (str): Path to energy dataset.
        weather_path (str): Path to weather dataset.

    Returns:
        bool: True if files exist, False otherwise.

    Raises:
        FileNotFoundError: If required files are missing.
    """
    logger = get_run_logger()

    energy_file = Path(energy_path)
    weather_file = Path(weather_path)

    if not energy_file.exists():
        raise FileNotFoundError(f"Energy dataset not found at {energy_path}")
    if not weather_file.exists():
        raise FileNotFoundError(f"Weather dataset not found at {weather_path}")

    logger.info(f"Data validation successful. Energy file size: {energy_file.stat().st_size} bytes")
    logger.info(f"Weather file size: {weather_file.stat().st_size} bytes")

    return True


@task(retries=2, retry_delay_seconds=30)
def run_zenml_pipeline(energy_path: str, weather_path: str, target_column: str = 'total_load_actual') -> Dict[str, Any]:
    """
    Execute the ZenML training pipeline with MLflow integration.

    Args:
        energy_path (str): Path to energy dataset.
        weather_path (str): Path to weather dataset.
        target_column (str): Target column for forecasting.

    Returns:
        Dict[str, Any]: Pipeline execution results.
    """
    logger = get_run_logger()

    try:
        # Import ZenML pipeline
        import sys
        sys.path.append('.')
        from zenml_pipelines.training_pipeline import energy_demand_training_pipeline

        logger.info("Starting ZenML pipeline execution with MLflow tracking...")

        # Run the pipeline
        energy_demand_training_pipeline(
            energy_path=energy_path,
            weather_path=weather_path,
            target_column=target_column
        )

        logger.info("ZenML pipeline execution completed successfully")

        # Retrieve results from ZenML (simplified - in practice you'd query ZenML client)
        return {
            'status': 'success',
            'pipeline_run_id': f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            'execution_time': datetime.now().isoformat(),
            'message': 'Pipeline executed successfully with MLflow tracking',
            'mlflow_experiment': 'energy_demand_forecasting'
        }

    except Exception as e:
        logger.error(f"ZenML pipeline execution failed: {str(e)}")
        raise


@task
def generate_training_report(pipeline_results: Dict[str, Any], output_dir: str = 'reports') -> str:
    """
    Generate a training report from pipeline results.

    Args:
        pipeline_results (Dict[str, Any]): Results from ZenML pipeline.
        output_dir (str): Directory to save the report.

    Returns:
        str: Path to the generated report.
    """
    logger = get_run_logger()

    # Ensure output directory exists
    report_dir = Path(output_dir)
    report_dir.mkdir(parents=True, exist_ok=True)

    # Generate report filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = report_dir / f"training_report_{timestamp}.txt"

    try:
        with open(report_path, 'w') as f:
            f.write("Energy Demand Forecasting - Training Report\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

            if pipeline_results.get('status') == 'success':
                f.write("Pipeline Status: SUCCESS\n\n")
                f.write("Execution Details:\n")
                f.write(f"- Pipeline Run ID: {pipeline_results.get('pipeline_run_id', 'N/A')}\n")
                f.write(f"- Execution Time: {pipeline_results.get('execution_time', 'N/A')}\n")
                f.write("- Message: Pipeline executed successfully\n\n")

                # Placeholder for model metrics (would be populated from ZenML)
                f.write("Model Performance Summary:\n")
                f.write("- Best Model: To be determined from ZenML artifacts\n")
                f.write("- Metrics: To be retrieved from ZenML experiment tracking\n\n")

            else:
                f.write("Pipeline Status: FAILED\n\n")
                f.write(f"Error: {pipeline_results.get('error', 'Unknown error')}\n\n")

            f.write("Next Steps:\n")
            f.write("- Review model performance metrics\n")
            f.write("- Deploy best model to production\n")
            f.write("- Set up monitoring and alerting\n")

        logger.info(f"Training report generated: {report_path}")

        return str(report_path)

    except Exception as e:
        logger.error(f"Failed to generate training report: {str(e)}")
        raise


@task
def send_notification(message: str, notification_type: str = 'info') -> bool:
    """
    Send notification about pipeline execution status.

    Args:
        message (str): Notification message.
        notification_type (str): Type of notification ('info', 'success', 'error').

    Returns:
        bool: True if notification sent successfully.
    """
    logger = get_run_logger()

    # In a real implementation, this would integrate with email, Slack, etc.
    # For now, just log the notification
    logger.info(f"[{notification_type.upper()}] {message}")

    # Placeholder for actual notification logic
    # Example integrations:
    # - Send email via SMTP
    # - Send Slack message via webhook
    # - Send Teams message via webhook
    # - Send SMS via Twilio

    return True


@flow(name="Energy Demand Forecasting Orchestration", description="Complete ML pipeline orchestration with ZenML and monitoring")
def energy_demand_orchestration_flow(
    energy_path: str = 'data/raw/energy_dataset.csv',
    weather_path: str = 'data/raw/weather_features.csv',
    target_column: str = 'total_load_actual',
    enable_notifications: bool = True
) -> Dict[str, Any]:
    """
    Main orchestration flow for energy demand forecasting.

    Args:
        energy_path (str): Path to energy dataset.
        weather_path (str): Path to weather dataset.
        target_column (str): Target column for forecasting.
        enable_notifications (bool): Whether to send notifications.

    Returns:
        Dict[str, Any]: Flow execution results.
    """
    logger = get_run_logger()
    start_time = datetime.now()

    logger.info("Starting Energy Demand Forecasting Orchestration Flow")
    logger.info(f"Parameters: energy_path={energy_path}, weather_path={weather_path}, target_column={target_column}")

    try:
        # Step 1: Validate data availability
        logger.info("Step 1: Validating data availability...")
        data_valid = validate_data_availability(energy_path, weather_path)

        # Step 2: Run ZenML training pipeline
        logger.info("Step 2: Executing ZenML training pipeline...")
        pipeline_results = run_zenml_pipeline(energy_path, weather_path, target_column)

        # Step 3: Generate training report
        logger.info("Step 3: Generating training report...")
        report_path = generate_training_report(pipeline_results)

        # Step 4: Send success notification
        if enable_notifications:
            success_message = f"Energy demand forecasting pipeline completed successfully. Report: {report_path}"
            send_notification(success_message, 'success')

        end_time = datetime.now()
        execution_time = (end_time - start_time).total_seconds()

        results = {
            'status': 'success',
            'start_time': start_time.isoformat(),
            'end_time': end_time.isoformat(),
            'execution_time_seconds': execution_time,
            'pipeline_results': pipeline_results,
            'report_path': report_path,
            'data_validated': data_valid
        }

        logger.info(f"Flow completed successfully in {execution_time:.2f} seconds")
        return results

    except Exception as e:
        error_message = f"Flow failed with error: {str(e)}"
        logger.error(error_message)

        # Send failure notification
        if enable_notifications:
            send_notification(error_message, 'error')

        # Re-raise to mark flow as failed
        raise


@flow(name="Scheduled Energy Demand Training", description="Scheduled execution of energy demand forecasting pipeline")
def scheduled_training_flow(
    schedule_interval: str = "daily",
    energy_path: str = 'data/raw/energy_dataset.csv',
    weather_path: str = 'data/raw/weather_features.csv',
    target_column: str = 'total_load_actual'
):
    """
    Scheduled flow for regular model retraining.

    Args:
        schedule_interval (str): Scheduling interval ('daily', 'weekly', 'monthly').
        energy_path (str): Path to energy dataset.
        weather_path (str): Path to weather dataset.
        target_column (str): Target column for forecasting.
    """
    logger = get_run_logger()

    logger.info(f"Starting scheduled training flow with interval: {schedule_interval}")

    # Execute main orchestration flow
    results = energy_demand_orchestration_flow(
        energy_path=energy_path,
        weather_path=weather_path,
        target_column=target_column,
        enable_notifications=True
    )

    logger.info("Scheduled training flow completed")
    return results


if __name__ == "__main__":
    # Example usage
    import argparse

    parser = argparse.ArgumentParser(description='Run Prefect orchestration flow')
    parser.add_argument('--energy-path', type=str, default='data/raw/energy_dataset.csv',
                       help='Path to energy dataset')
    parser.add_argument('--weather-path', type=str, default='data/raw/weather_features.csv',
                       help='Path to weather dataset')
    parser.add_argument('--target-column', type=str, default='total_load_actual',
                       help='Target column for forecasting')
    parser.add_argument('--scheduled', action='store_true',
                       help='Run as scheduled flow')
    parser.add_argument('--disable-notifications', action='store_true',
                       help='Disable notifications')

    args = parser.parse_args()

    if args.scheduled:
        # Run scheduled flow
        scheduled_training_flow(
            energy_path=args.energy_path,
            weather_path=args.weather_path,
            target_column=args.target_column
        )
    else:
        # Run main orchestration flow
        energy_demand_orchestration_flow(
            energy_path=args.energy_path,
            weather_path=args.weather_path,
            target_column=args.target_column,
            enable_notifications=not args.disable_notifications
        )

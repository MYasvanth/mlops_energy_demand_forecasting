mkdir artifacts
mkdir configs
mkdir configs\data
mkdir configs\deployment
mkdir configs\experiment
mkdir configs\model
mkdir configs\monitoring
mkdir configs\schemas
mkdir configs\training
mkdir data
mkdir data\external
mkdir data\monitoring
mkdir data\processed
mkdir data\raw
mkdir deployment
mkdir deployment\cloud
mkdir deployment\cloud\aws
mkdir deployment\cloud\azure
mkdir deployment\cloud\gcp
mkdir deployment\docker
mkdir deployment\kubernetes
mkdir docs
mkdir docs\diagrams
mkdir logs
mkdir mlartifacts
mkdir models
mkdir models\archived
mkdir models\encoders
mkdir models\production
mkdir models\staging
mkdir monitoring
mkdir monitoring\alerts
mkdir monitoring\dashboards
mkdir monitoring\logs
mkdir notebooks
mkdir reports
mkdir reports\data_quality
mkdir reports\drift_reports
mkdir reports\hyperparameter_tuning
mkdir reports\model_performance
mkdir reports\monitoring_reports
mkdir reports\performance_reports
mkdir scripts
mkdir scripts\deployment
mkdir scripts\ingestion
mkdir scripts\monitoring
mkdir scripts\testing
mkdir scripts\training
mkdir scripts\validation
mkdir src
mkdir src\data
mkdir src\deployment
mkdir src\features
mkdir src\models
mkdir src\monitoring
mkdir src\pipelines
mkdir src\utils
mkdir tests
mkdir tests\test_data
mkdir tests\test_deployment
mkdir tests\test_features
mkdir tests\test_integration
mkdir tests\test_models
mkdir tests\test_monitoring
mkdir tests\test_performance
mkdir zenml_pipelines

REM Create empty files
echo. > __init__.py
echo. > .dvcignore
echo. > .gitignore
echo. > DEPLOYMENT_COMPLETION_SUMMARY.md
echo. > DEPLOYMENT_GUIDE.md
echo. > DOCKER_SETUP_GUIDE.md
echo. > dvc.yaml
echo. > EXECUTION_STEPS.md
echo. > MLFLOW_AUTOMATION_SUMMARY.md
echo. > MODEL_ALIGNMENT_SUMMARY.md
echo. > MONITORING_README.md
echo. > params.yaml
echo. > README.md
echo. > requirements.txt
echo. > ROOT_FILES_ANALYSIS.md
echo. > run_demand_pipeline.py
echo. > SETUP_DOCKER_WINDOWS.md
echo. > setup.py
echo. > streamlit_requirements.txt
echo. > TODO_EVIDENTLY_INTEGRATION.md
echo. > TODO.md

REM Create __init__.py in subdirs
echo. > src\__init__.py
echo. > tests\__init__.py
echo. > zenml_pipelines\__init__.py

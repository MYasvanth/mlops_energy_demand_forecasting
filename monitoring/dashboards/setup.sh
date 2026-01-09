#!/bin/bash
# Setup script for Streamlit Cloud deployment with DVC

# Set DVC environment variables from Streamlit secrets
export DVC_S3_ACCESS_KEY_ID="$DVC_ACCESS_KEY_ID"
export DVC_S3_SECRET_ACCESS_KEY="$DVC_SECRET_ACCESS_KEY"

# Initialize DVC if not already done
if [ ! -d ".dvc" ]; then
    dvc init --no-scm
fi

# Configure remote if not exists
dvc remote add -d s3remote s3://your-bucket/mlops-energy-data/ -f

# Pull data
dvc pull
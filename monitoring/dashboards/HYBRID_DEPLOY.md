# Hybrid Deployment: DVC + GitHub Fallback

## Data Loading Strategy:
1. **DVC Pull** (if available and configured)
2. **Local Files** (if DVC pulled successfully)
3. **GitHub URLs** (fallback if DVC fails)

## Setup Options:

### Option A: DVC with Cloud Storage
```bash
# Configure DVC remote (choose one):
dvc remote add -d s3remote s3://your-bucket/data/
dvc remote add -d gcs gs://your-bucket/data/
dvc remote add -d azure azure://container/data/

# Push data
dvc push
```

**Streamlit Secrets:**
```toml
# For AWS S3
AWS_ACCESS_KEY_ID = "your-key"
AWS_SECRET_ACCESS_KEY = "your-secret"

# For Google Cloud
GOOGLE_APPLICATION_CREDENTIALS = "path-to-service-account.json"
```

### Option B: GitHub Only (No Setup)
- Data automatically loads from GitHub URLs
- No credentials needed
- Works immediately

## Benefits:
- ✅ **Reliable**: GitHub fallback ensures data always loads
- ✅ **Flexible**: Use DVC for large datasets, GitHub for simple deployment
- ✅ **Fast**: DVC for local development, GitHub for cloud
- ✅ **Secure**: Credentials only needed for DVC, not GitHub

## Deploy to Streamlit Cloud:
1. Repository: `MYasvanth/mlops_energy_demand_forecasting`
2. Main file: `monitoring/dashboards/prediction_dashboard_lite.py`
3. Add secrets (optional, for DVC)
4. Deploy
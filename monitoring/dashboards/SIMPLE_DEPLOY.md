# Simple Streamlit Cloud Deployment

## Quick Setup (No DVC needed):

### 1. Commit data files to Git
```bash
# Remove from .gitignore
git rm --cached data/raw/*.csv
git add data/raw/energy_dataset.csv data/raw/weather_features.csv
git commit -m "Add data files for Streamlit deployment"
git push
```

### 2. Update GitHub URL in data_loader.py
Replace `your-username` with your actual GitHub username:
```python
github_url = "https://raw.githubusercontent.com/YOUR-USERNAME/mlops_energy_demand_forecasting/main/data/raw/energy_dataset.csv"
```

### 3. Deploy to Streamlit Cloud
1. Go to https://share.streamlit.io/
2. Connect GitHub repository
3. Set main file: `monitoring/dashboards/prediction_dashboard_lite.py`
4. Deploy

## How it works:
- Loads data directly from GitHub raw URLs
- No authentication needed
- Fast and simple deployment
- Data cached by Streamlit

## Files needed:
- `prediction_dashboard_lite.py` (main app)
- `data_loader.py` (simplified)
- `requirements.txt` (minimal dependencies)
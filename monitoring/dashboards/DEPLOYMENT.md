# Streamlit Cloud Deployment with DVC + Google Drive (Free)

## Setup Steps:

### 1. Configure DVC with Google Drive (Local)
```bash
# Initialize DVC
dvc init

# Add Google Drive remote (replace with your folder ID)
dvc remote add -d gdrive gdrive://1BxiMVs0XRA5nFMdKvBdBZjgmUUqptlbs74OgvE2upms

# Add data files to DVC
dvc add data/raw/energy_dataset.csv
dvc add data/raw/weather_features.csv

# Push to Google Drive (will prompt for Google auth)
dvc push
```

### 2. Commit DVC files to Git
```bash
git add .dvc/config data/raw/*.dvc .gitignore
git commit -m "Add DVC configuration and data tracking"
git push
```

### 3. Deploy to Streamlit Cloud
1. Go to https://share.streamlit.io/
2. Connect your GitHub repository
3. Set main file path: `monitoring/dashboards/prediction_dashboard_lite.py`
4. Deploy

### 4. Google Drive Folder Setup
1. Create a folder in Google Drive
2. Get folder ID from URL: `https://drive.google.com/drive/folders/FOLDER_ID`
3. Update `.dvc/config` with your folder ID
4. Make folder publicly accessible (View permissions)

## How it Works:
- DVC pulls data from Google Drive on first app load
- Data is cached by Streamlit for subsequent requests
- No authentication required (public folder)
- Free storage up to 15GB per Google account

## Files Structure:
```
monitoring/dashboards/
├── prediction_dashboard_lite.py  # Main app
├── data_loader.py                # Data loading logic
├── requirements.txt              # Dependencies
└── .dvc/config                   # DVC configuration
```
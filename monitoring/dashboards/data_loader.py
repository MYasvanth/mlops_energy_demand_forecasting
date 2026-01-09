"""
Data loader for Streamlit Cloud deployment
Handles multiple data loading strategies based on environment
"""

import os
import pandas as pd
import streamlit as st
import subprocess
import logging
from pathlib import Path
from typing import Optional, Tuple

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataLoader:
    """Handles data loading for different deployment environments."""
    
    def __init__(self):
        self.data_dir = Path("data")
        self.raw_dir = self.data_dir / "raw"
        self.processed_dir = self.data_dir / "processed"
        
    def is_streamlit_cloud(self) -> bool:
        """Check if running on Streamlit Cloud."""
        return os.getenv("STREAMLIT_SHARING_MODE") is not None or \
               "streamlit.io" in os.getenv("HOSTNAME", "")
    
    def check_dvc_available(self) -> bool:
        """Check if DVC is available and configured."""
        try:
            result = subprocess.run(["dvc", "version"], 
                                  capture_output=True, text=True, timeout=10)
            return result.returncode == 0
        except (subprocess.TimeoutExpired, FileNotFoundError):
            return False
    
    @st.cache_data
    def pull_dvc_data(_self) -> bool:
        """Pull data using DVC with Google Drive."""
        try:
            logger.info("Pulling data with DVC from Google Drive...")
            result = subprocess.run(["dvc", "pull"], 
                                  capture_output=True, text=True, timeout=300)
            if result.returncode == 0:
                logger.info("DVC pull successful")
                return True
            else:
                logger.warning(f"DVC pull failed: {result.stderr}")
                return False
        except subprocess.TimeoutExpired:
            logger.error("DVC pull timed out")
            return False
        except Exception as e:
            logger.error(f"DVC pull error: {e}")
            return False
    
    @st.cache_data
    def load_from_url(_self, url: str) -> Optional[pd.DataFrame]:
        """Load data from URL as fallback."""
        try:
            logger.info(f"Loading data from URL: {url}")
            return pd.read_csv(url)
        except Exception as e:
            logger.error(f"Failed to load from URL {url}: {e}")
            return None
    

    
    def load_energy_data(self) -> Optional[pd.DataFrame]:
        """Load energy dataset from GitHub raw URL."""
        # GitHub raw URL for energy dataset
        github_url = "https://raw.githubusercontent.com/MYasvanth/mlops_energy_demand_forecasting/main/data/raw/energy_dataset.csv"

        try:
            logger.info("Loading energy data from GitHub")
            return pd.read_csv(github_url)
        except Exception as e:
            logger.error(f"Failed to load from GitHub: {e}")
            return None
    
    def load_weather_data(self) -> Optional[pd.DataFrame]:
        """Load weather dataset with fallback strategies."""
        weather_file = self.raw_dir / "weather_features.csv"
        
        # Strategy 1: Try local file first
        if weather_file.exists():
            try:
                logger.info("Loading weather data from local file")
                return pd.read_csv(weather_file)
            except Exception as e:
                logger.warning(f"Failed to load local weather file: {e}")
        
        # Strategy 2: Try DVC pull if available
        if self.check_dvc_available():
            if self.pull_dvc_data() and weather_file.exists():
                try:
                    return pd.read_csv(weather_file)
                except Exception as e:
                    logger.warning(f"Failed to load weather after DVC pull: {e}")
        
        # Strategy 3: Try loading from cloud storage URL (if configured)
        cloud_url = st.secrets.get("WEATHER_DATA_URL") if hasattr(st, 'secrets') else None
        if cloud_url:
            data = self.load_from_url(cloud_url)
            if data is not None:
                return data
        
        # No weather data available
        logger.info("No weather data available")
        return None
    
    def load_processed_data(self) -> Optional[pd.DataFrame]:
        """Load processed dataset if available."""
        processed_file = self.processed_dir / "processed_energy_weather.csv"
        
        if processed_file.exists():
            try:
                logger.info("Loading processed data from local file")
                return pd.read_csv(processed_file)
            except Exception as e:
                logger.warning(f"Failed to load processed file: {e}")
        
        return None
    
    def get_data_info(self) -> dict:
        """Get information about available data sources."""
        info = {
            "environment": "Streamlit Cloud" if self.is_streamlit_cloud() else "Local",
            "dvc_available": self.check_dvc_available(),
            "local_files": {
                "energy_dataset": (self.raw_dir / "energy_dataset.csv").exists(),
                "weather_features": (self.raw_dir / "weather_features.csv").exists(),
                "processed_data": (self.processed_dir / "processed_energy_weather.csv").exists()
            }
        }
        return info

# Global instance
data_loader = DataLoader()

@st.cache_data
def load_energy_dataset():
    """Load energy dataset with fallback strategies."""
    return data_loader.load_energy_data()

@st.cache_data
def load_weather_dataset():
    """Load weather dataset with fallback strategies."""
    return data_loader.load_weather_data()
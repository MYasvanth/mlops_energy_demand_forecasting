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
        """Pull data using DVC with Google Drive or other remotes."""
        try:
            # Set credentials from Streamlit secrets if available
            if hasattr(st, 'secrets'):
                # Google Cloud
                if 'GOOGLE_APPLICATION_CREDENTIALS' in st.secrets:
                    os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = st.secrets['GOOGLE_APPLICATION_CREDENTIALS']
                # AWS
                if 'AWS_ACCESS_KEY_ID' in st.secrets:
                    os.environ['AWS_ACCESS_KEY_ID'] = st.secrets['AWS_ACCESS_KEY_ID']
                    os.environ['AWS_SECRET_ACCESS_KEY'] = st.secrets['AWS_SECRET_ACCESS_KEY']
            
            logger.info("Pulling data with DVC...")
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
        """Load energy dataset with DVC and GitHub fallback."""
        energy_file = self.raw_dir / "energy_dataset.csv"
        
        # Strategy 1: Try DVC pull if available
        if self.check_dvc_available():
            if self.pull_dvc_data() and energy_file.exists():
                try:
                    logger.info("Loading energy data from DVC")
                    return pd.read_csv(energy_file)
                except Exception as e:
                    logger.warning(f"Failed to load after DVC pull: {e}")
        
        # Strategy 2: Try local file
        if energy_file.exists():
            try:
                logger.info("Loading energy data from local file")
                return pd.read_csv(energy_file)
            except Exception as e:
                logger.warning(f"Failed to load local file: {e}")
        
        # Strategy 3: GitHub raw URL fallback
        github_url = "https://raw.githubusercontent.com/MYasvanth/mlops_energy_demand_forecasting/main/data/raw/energy_dataset.csv"
        try:
            logger.info("Loading energy data from GitHub")
            return pd.read_csv(github_url)
        except Exception as e:
            logger.error(f"Failed to load from GitHub: {e}")
            return None
    
    def load_weather_data(self) -> Optional[pd.DataFrame]:
        """Load weather dataset with DVC and GitHub fallback."""
        weather_file = self.raw_dir / "weather_features.csv"
        
        # Strategy 1: Try DVC pull if available
        if self.check_dvc_available():
            if self.pull_dvc_data() and weather_file.exists():
                try:
                    logger.info("Loading weather data from DVC")
                    return pd.read_csv(weather_file)
                except Exception as e:
                    logger.warning(f"Failed to load weather after DVC pull: {e}")
        
        # Strategy 2: Try local file
        if weather_file.exists():
            try:
                logger.info("Loading weather data from local file")
                return pd.read_csv(weather_file)
            except Exception as e:
                logger.warning(f"Failed to load local weather file: {e}")
        
        # Strategy 3: GitHub raw URL fallback
        github_url = "https://raw.githubusercontent.com/MYasvanth/mlops_energy_demand_forecasting/main/data/raw/weather_features.csv"
        try:
            logger.info("Loading weather data from GitHub")
            return pd.read_csv(github_url)
        except Exception as e:
            logger.error(f"Failed to load weather from GitHub: {e}")
            return None
    
    def load_processed_data(self) -> Optional[pd.DataFrame]:
        """Load processed dataset with DVC and GitHub fallback."""
        processed_file = self.processed_dir / "processed_energy_weather.csv"
        
        # Strategy 1: Try DVC pull if available
        if self.check_dvc_available():
            if self.pull_dvc_data() and processed_file.exists():
                try:
                    logger.info("Loading processed data from DVC")
                    return pd.read_csv(processed_file)
                except Exception as e:
                    logger.warning(f"Failed to load processed after DVC pull: {e}")
        
        # Strategy 2: Try local file
        if processed_file.exists():
            try:
                logger.info("Loading processed data from local file")
                return pd.read_csv(processed_file)
            except Exception as e:
                logger.warning(f"Failed to load processed file: {e}")
        
        # Strategy 3: GitHub raw URL fallback
        github_url = "https://raw.githubusercontent.com/MYasvanth/mlops_energy_demand_forecasting/main/data/processed/processed_energy_weather.csv"
        try:
            logger.info("Loading processed data from GitHub")
            return pd.read_csv(github_url)
        except Exception as e:
            logger.warning(f"Failed to load processed from GitHub: {e}")
        
        # Strategy 4: Generate from raw data
        try:
            energy_data = self.load_energy_data()
            if energy_data is not None:
                logger.info("Using energy data as processed data")
                return energy_data
        except Exception as e:
            logger.error(f"Failed to create processed data: {e}")
        
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
    """Load energy dataset from GitHub."""
    return data_loader.load_energy_data()

@st.cache_data
def load_weather_dataset():
    """Load weather dataset from GitHub."""
    return data_loader.load_weather_data()

@st.cache_data
def load_processed_dataset():
    """Load processed dataset."""
    return data_loader.load_processed_data()
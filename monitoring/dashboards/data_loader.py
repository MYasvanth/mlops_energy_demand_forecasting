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
        """Skip DVC pull for free deployment."""
        logger.info("Skipping DVC pull - using local/GitHub data only")
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
        """Load processed dataset without cloud storage."""
        processed_file = self.processed_dir / "processed_energy_weather.csv"
        
        # Strategy 1: Try local file first
        if processed_file.exists():
            try:
                logger.info("Loading processed data from local file")
                return pd.read_csv(processed_file)
            except Exception as e:
                logger.warning(f"Failed to load processed file: {e}")
        
        # Strategy 2: Generate from raw data (no DVC needed)
        try:
            energy_data = self.load_energy_data()
            weather_data = self.load_weather_data()
            
            if energy_data is not None:
                # Simple processing: use energy data as base
                logger.info("Generating processed data from raw energy data")
                processed_data = energy_data.copy()
                
                # Add weather features if available
                if weather_data is not None and len(weather_data) == len(energy_data):
                    weather_cols = ['temperature', 'humidity', 'wind_speed']
                    for col in weather_cols:
                        if col in weather_data.columns:
                            processed_data[col] = weather_data[col]
                
                return processed_data
                
        except Exception as e:
            logger.error(f"Failed to generate processed data: {e}")
        
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

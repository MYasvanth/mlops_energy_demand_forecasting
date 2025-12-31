"""
Data Ingestion Module

This module handles loading and validating data from various sources such as CSV files, APIs, and databases.
It uses Pydantic for data validation to ensure data integrity and type safety.
"""

import logging
from typing import Dict, List, Optional, Union
import pandas as pd
from pydantic import BaseModel, Field, ValidationError
from sqlalchemy import create_engine, text
import requests

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Pydantic models for data validation
class EnergyDataRecord(BaseModel):
    """
    Pydantic model for validating energy generation data records.
    """
    time: str = Field(..., description="Timestamp of the record", min_length=1)
    generation_biomass: Optional[float] = Field(None, description="Biomass generation")
    generation_fossil_brown_coal_lignite: Optional[float] = Field(None, description="Fossil brown coal lignite generation")
    generation_fossil_coal_derived_gas: Optional[float] = Field(None, description="Fossil coal derived gas generation")
    generation_fossil_gas: Optional[float] = Field(None, description="Fossil gas generation")
    generation_fossil_hard_coal: Optional[float] = Field(None, description="Fossil hard coal generation")
    generation_fossil_oil: Optional[float] = Field(None, description="Fossil oil generation")
    generation_hydro_pumped_storage_consumption: Optional[float] = Field(None, description="Hydro pumped storage consumption")
    generation_hydro_run_of_river_and_poundage: Optional[float] = Field(None, description="Hydro run of river and poundage generation")
    generation_hydro_water_reservoir: Optional[float] = Field(None, description="Hydro water reservoir generation")
    generation_nuclear: Optional[float] = Field(None, description="Nuclear generation")
    generation_other: Optional[float] = Field(None, description="Other generation")
    generation_other_renewable: Optional[float] = Field(None, description="Other renewable generation")
    generation_solar: Optional[float] = Field(None, description="Solar generation")
    generation_waste: Optional[float] = Field(None, description="Waste generation")
    generation_wind_onshore: Optional[float] = Field(None, description="Wind onshore generation")
    forecast_solar_day_ahead: Optional[float] = Field(None, description="Solar day ahead forecast")
    forecast_wind_onshore_day_ahead: Optional[float] = Field(None, description="Wind onshore day ahead forecast")
    total_load_forecast: Optional[float] = Field(None, description="Total load forecast")
    total_load_actual: Optional[float] = Field(None, description="Total load actual")
    price_day_ahead: Optional[float] = Field(None, description="Day ahead price")
    price_actual: Optional[float] = Field(None, description="Actual price")


class WeatherDataRecord(BaseModel):
    """
    Pydantic model for validating weather features data records.
    """
    dt_iso: str = Field(..., description="ISO datetime")
    city_name: Optional[str] = Field(None, description="City name")
    temp: Optional[float] = Field(None, description="Temperature")
    temp_min: Optional[float] = Field(None, description="Minimum temperature")
    temp_max: Optional[float] = Field(None, description="Maximum temperature")
    pressure: Optional[float] = Field(None, description="Pressure")
    humidity: Optional[float] = Field(None, description="Humidity")
    wind_speed: Optional[float] = Field(None, description="Wind speed")
    wind_deg: Optional[float] = Field(None, description="Wind degree")
    rain_1h: Optional[float] = Field(None, description="Rain in last 1 hour")
    rain_3h: Optional[float] = Field(None, description="Rain in last 3 hours")
    snow_1h: Optional[float] = Field(None, description="Snow in last 1 hour")
    snow_3h: Optional[float] = Field(None, description="Snow in last 3 hours")
    clouds_all: Optional[float] = Field(None, description="Cloudiness percentage")
    weather_id: Optional[int] = Field(None, description="Weather condition id")
    weather_main: Optional[str] = Field(None, description="Weather main")
    weather_description: Optional[str] = Field(None, description="Weather description")
    weather_icon: Optional[str] = Field(None, description="Weather icon")


def load_csv_data(file_path: str, model: BaseModel, sep: str = ',') -> pd.DataFrame:
    """
    Load data from a CSV file and validate each record using a Pydantic model with robust error handling.

    Args:
        file_path (str): Path to the CSV file.
        model (BaseModel): Pydantic model for validation.
        sep (str): Separator for CSV file. Defaults to ','.

    Returns:
        pd.DataFrame: Validated DataFrame.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValidationError: If data validation fails.
        Exception: For other errors during loading.
    """
    try:
        logger.info(f"Loading data from {file_path}")

        # Check if file exists
        if not pd.io.common.file_exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")

        # Try different separators if default fails
        separators = [sep, ';', '\t']
        df = None
        successful_sep = sep

        for sep_try in separators:
            try:
                df = pd.read_csv(file_path, sep=sep_try, low_memory=False)
                if len(df.columns) > 1:  # Ensure we have multiple columns
                    successful_sep = sep_try
                    break
            except Exception:
                continue

        if df is None or df.empty:
            raise ValueError(f"Could not load data from {file_path} with any separator")

        logger.info(f"Successfully loaded {len(df)} records using separator '{successful_sep}'")

        # Clean column names: replace spaces with underscores and convert to lowercase
        df.columns = df.columns.str.replace(' ', '_').str.lower().str.replace(r'[^\w]', '_', regex=True)
        logger.info(f"Column names cleaned: {list(df.columns)}")

        # Basic data quality checks
        if df.empty:
            raise ValueError("Loaded DataFrame is empty")

        if len(df.columns) < 2:
            raise ValueError("DataFrame has insufficient columns")

        # Validate each row with error tolerance
        validated_records = []
        validation_errors = 0
        max_errors = min(100, len(df) // 10)  # Allow up to 10% validation errors

        for index, row in df.iterrows():
            try:
                # Convert row to dict and handle NaN values
                row_dict = {}
                for col, val in row.items():
                    if pd.isna(val):
                        row_dict[col] = None
                    else:
                        row_dict[col] = val

                record = model(**row_dict)
                validated_records.append(record.model_dump())
            except ValidationError as e:
                validation_errors += 1
                if validation_errors <= 5:  # Log first few errors
                    logger.warning(f"Validation error at row {index}: {e}")
                if validation_errors > max_errors:
                    logger.error(f"Too many validation errors ({validation_errors}). Stopping validation.")
                    break
                continue  # Skip invalid rows

        if not validated_records:
            raise ValueError("No valid records found after validation")

        validated_df = pd.DataFrame(validated_records)
        logger.info(f"Successfully validated {len(validated_df)} records out of {len(df)} total (skipped {validation_errors} invalid records)")
        return validated_df

    except FileNotFoundError:
        logger.error(f"File not found: {file_path}")
        raise
    except ValidationError as e:
        logger.error(f"Data validation failed: {e}")
        raise
    except pd.errors.EmptyDataError:
        logger.error(f"Empty CSV file: {file_path}")
        raise ValueError(f"CSV file is empty: {file_path}")
    except Exception as e:
        logger.error(f"Error loading CSV data: {str(e)}")
        raise


def load_energy_data(file_path: str) -> pd.DataFrame:
    """
    Load and validate energy dataset from CSV.

    Args:
        file_path (str): Path to the energy dataset CSV file.

    Returns:
        pd.DataFrame: Validated energy data DataFrame.
    """
    return load_csv_data(file_path, EnergyDataRecord)


def load_weather_data(file_path: str) -> pd.DataFrame:
    """
    Load and validate weather features from CSV.

    Args:
        file_path (str): Path to the weather features CSV file.

    Returns:
        pd.DataFrame: Validated weather data DataFrame.
    """
    return load_csv_data(file_path, WeatherDataRecord)


def load_from_api(api_url: str, params: Optional[Dict[str, Union[str, int]]] = None) -> pd.DataFrame:
    """
    Load data from an API endpoint.

    Args:
        api_url (str): URL of the API endpoint.
        params (Optional[Dict[str, Union[str, int]]]): Query parameters for the API.

    Returns:
        pd.DataFrame: DataFrame from API response.

    Raises:
        requests.RequestException: If the API request fails.
    """
    try:
        logger.info(f"Fetching data from API: {api_url}")
        response = requests.get(api_url, params=params)
        response.raise_for_status()
        data = response.json()
        df = pd.DataFrame(data)
        logger.info(f"Loaded {len(df)} records from API")
        return df
    except requests.RequestException as e:
        logger.error(f"API request failed: {e}")
        raise


def load_from_database(db_url: str, query: str) -> pd.DataFrame:
    """
    Load data from a database using SQL query.

    Args:
        db_url (str): Database connection URL.
        query (str): SQL query to execute.

    Returns:
        pd.DataFrame: DataFrame from query results.

    Raises:
        Exception: If database connection or query fails.
    """
    try:
        logger.info(f"Connecting to database: {db_url}")
        engine = create_engine(db_url)
        with engine.connect() as conn:
            df = pd.read_sql(text(query), conn)
        logger.info(f"Loaded {len(df)} records from database")
        return df
    except Exception as e:
        logger.error(f"Database query failed: {e}")
        raise


def merge_datasets(energy_df: pd.DataFrame, weather_df: pd.DataFrame) -> pd.DataFrame:
    """
    Merge energy and weather datasets on timestamp.

    Args:
        energy_df (pd.DataFrame): Energy data DataFrame.
        weather_df (pd.DataFrame): Weather data DataFrame.

    Returns:
        pd.DataFrame: Merged DataFrame.
    """
    # Convert timestamps
    energy_df['time'] = pd.to_datetime(energy_df['time'])
    weather_df['dt_iso'] = pd.to_datetime(weather_df['dt_iso'])

    # Rename columns for consistency
    weather_df = weather_df.rename(columns={'dt_iso': 'time'})

    # Merge on time
    merged_df = pd.merge(energy_df, weather_df, on='time', how='inner')
    logger.info(f"Merged data shape: {merged_df.shape}")
    return merged_df


def ingest_data(energy_file: str, weather_file: str) -> Dict[str, pd.DataFrame]:
    """
    Ingest energy and weather data from CSV files.

    Args:
        energy_file (str): Path to energy dataset CSV.
        weather_file (str): Path to weather features CSV.

    Returns:
        Dict[str, pd.DataFrame]: Dictionary containing 'energy' and 'weather' DataFrames.
    """
    energy_df = load_energy_data(energy_file)
    weather_df = load_weather_data(weather_file)
    return {'energy': energy_df, 'weather': weather_df}


if __name__ == "__main__":
    # Example usage
    energy_path = "data/raw/energy_dataset.csv"
    weather_path = "data/raw/weather_features.csv"
    data = ingest_data(energy_path, weather_path)
    print("Data ingestion completed successfully.")
    print(f"Energy data shape: {data['energy'].shape}")
    print(f"Weather data shape: {data['weather'].shape}")

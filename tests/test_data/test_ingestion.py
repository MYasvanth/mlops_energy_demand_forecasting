import pytest
import pandas as pd
from unittest.mock import patch, MagicMock
from src.data.ingestion import (
    load_csv_data,
    load_energy_data,
    load_weather_data,
    load_from_api,
    load_from_database,
    ingest_data,
    EnergyDataRecord,
    WeatherDataRecord
)


class TestLoadCsvData:
    def test_load_csv_data_success(self, tmp_path):
        # Create a temporary CSV file
        csv_content = """time,generation_biomass,generation_fossil_gas
2023-01-01 00:00:00,100.0,200.0
2023-01-01 01:00:00,110.0,210.0"""
        csv_file = tmp_path / "test_energy.csv"
        csv_file.write_text(csv_content)

        df = load_csv_data(str(csv_file), EnergyDataRecord)
        assert len(df) == 2
        assert df.iloc[0]['generation_biomass'] == 100.0

    def test_load_csv_data_file_not_found(self):
        with pytest.raises(FileNotFoundError):
            load_csv_data("nonexistent.csv", EnergyDataRecord)

    def test_load_csv_data_validation_error(self, tmp_path):
        # CSV with invalid data (empty time)
        csv_content = """time,generation_biomass
,100.0"""
        csv_file = tmp_path / "invalid.csv"
        csv_file.write_text(csv_content)

        # Should raise ValueError for no valid records
        with pytest.raises(ValueError, match="No valid records found after validation"):
            load_csv_data(str(csv_file), EnergyDataRecord)


class TestLoadEnergyData:
    def test_load_energy_data(self, tmp_path):
        csv_content = """time,generation_biomass,generation_fossil_gas
2023-01-01 00:00:00,100.0,200.0"""
        csv_file = tmp_path / "energy.csv"
        csv_file.write_text(csv_content)

        df = load_energy_data(str(csv_file))
        assert len(df) == 1
        assert 'generation_biomass' in df.columns


class TestLoadWeatherData:
    def test_load_weather_data(self, tmp_path):
        csv_content = """dt_iso,city_name,temp
2023-01-01 00:00:00+00:00,Madrid,15.0"""
        csv_file = tmp_path / "weather.csv"
        csv_file.write_text(csv_content)

        df = load_weather_data(str(csv_file))
        assert len(df) == 1
        assert 'temp' in df.columns


class TestLoadFromApi:
    @patch('src.data.ingestion.requests.get')
    def test_load_from_api_success(self, mock_get):
        mock_response = MagicMock()
        mock_response.json.return_value = [{'key': 'value'}]
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response

        df = load_from_api("http://example.com")
        assert len(df) == 1
        assert df.iloc[0]['key'] == 'value'

    @patch('src.data.ingestion.requests.get')
    def test_load_from_api_failure(self, mock_get):
        mock_get.side_effect = Exception("API error")
        with pytest.raises(Exception):
            load_from_api("http://example.com")


class TestLoadFromDatabase:
    @patch('src.data.ingestion.create_engine')
    def test_load_from_database_success(self, mock_create_engine):
        mock_engine = MagicMock()
        mock_conn = MagicMock()
        mock_conn.__enter__.return_value = mock_conn
        mock_conn.__exit__.return_value = None
        mock_create_engine.return_value = mock_engine
        mock_engine.connect.return_value.__enter__.return_value = mock_conn

        # Mock pd.read_sql
        with patch('src.data.ingestion.pd.read_sql') as mock_read_sql:
            mock_read_sql.return_value = pd.DataFrame({'col': [1, 2]})
            df = load_from_database("sqlite:///:memory:", "SELECT * FROM table")
            assert len(df) == 2

    @patch('src.data.ingestion.create_engine')
    def test_load_from_database_failure(self, mock_create_engine):
        mock_create_engine.side_effect = Exception("DB error")
        with pytest.raises(Exception):
            load_from_database("sqlite:///:memory:", "SELECT * FROM table")


class TestIngestData:
    def test_ingest_data(self, tmp_path):
        # Create energy CSV
        energy_csv = tmp_path / "energy.csv"
        energy_csv.write_text("""time,generation_biomass
2023-01-01 00:00:00,100.0""")

        # Create weather CSV
        weather_csv = tmp_path / "weather.csv"
        weather_csv.write_text("""dt_iso,temp
2023-01-01 00:00:00+00:00,15.0""")

        data = ingest_data(str(energy_csv), str(weather_csv))
        assert 'energy' in data
        assert 'weather' in data
        assert len(data['energy']) == 1
        assert len(data['weather']) == 1

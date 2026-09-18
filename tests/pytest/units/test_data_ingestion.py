import os

import pandas as pd
import pytest

from src.components import data_ingestion
from src.utils import load_config

config = load_config()

@pytest.fixture
def sample_features(tmp_path):

    sample_features_dataframe = pd.DataFrame({
        'Date': ['2013-06-15', '2014-01-18', '2011-09-07', '2015-02-26', '2010-11-12',
                 '2016-03-09', '2013-12-03', '2012-07-21', '2015-10-29', '2014-05-16'],
        'Location': ['Melbourne', 'Sydney', 'Canberra', 'Brisbane', 'Perth',
                     'Adelaide', 'Darwin', 'Hobart', 'Cairns', 'Ballarat'],
        'MinTemp': [7.2, 19.4, 3.8, 23.1, 14.7, 16.3, 25.2, 5.4, 21.8, 8.6],
        'MaxTemp': [14.8, 29.7, 17.5, 31.8, 28.6, 27.4, 34.1, 13.2, 30.5, 18.7],
        'Rainfall': [2.6, 0.0, 0.0, 7.4, 0.0, 0.0, 12.8, 1.8, 3.2, 0.4],
        'Evaporation': [2.4, 8.2, 3.6, 7.9, 7.1, 6.8, 6.2, 1.9, 5.7, 3.1],
        'Sunshine': [4.5, 9.6, 8.9, 6.7, 11.2, 10.4, 5.3, 5.8, 7.4, 8.1],
        'WindGustDir': ['W', 'NE', 'NW', 'ESE', 'S', 'SW', 'NW', 'N', 'SE', 'WNW'],
        'WindGustSpeed': [41, 39, 37, 43, 35, 46, 52, 44, 40, 38],
        'WindDir9am': ['WSW', 'NNE', 'N', 'E', 'SE', 'NW', 'NW', 'NNW', 'SE', 'W'],
        'WindDir3pm': ['W', 'E', 'NW', 'ESE', 'SW', 'SW', 'W', 'N', 'ESE', 'WNW'],
        'WindSpeed9am': [17, 13, 11, 18, 13, 20, 24, 15, 16, 14],
        'WindSpeed3pm': [24, 22, 20, 27, 25, 29, 30, 23, 25, 21],
        'Humidity9am': [82, 67, 74, 76, 62, 58, 81, 79, 78, 71],
        'Humidity3pm': [68, 51, 42, 64, 38, 43, 69, 61, 67, 55],
        'Pressure9am': [1018.4, 1012.6, 1022.3, 1010.8, 1016.1,
                        1014.7, 1007.5, 1019.8, 1011.2, 1020.4],
        'Pressure3pm': [1016.2, 1010.4, 1018.7, 1008.3, 1013.5,
                        1012.1, 1005.8, 1017.2, 1009.6, 1018.9],
        'Cloud9am': [7, 2, 1, 5, 1, 2, 7, 5, 4, 3],
        'Cloud3pm': [6, 3, 2, 6, 1, 3, 8, 4, 5, 4],
        'Temp9am': [9.8, 23.1, 8.7, 26.4, 19.8, 20.7, 29.0, 8.1, 25.6, 12.4],
        'Temp3pm': [14.1, 28.4, 16.2, 30.1, 27.2, 26.1, 32.6, 12.4, 29.3, 17.1],
        'RainToday': ['Yes', 'No', 'No', 'Yes', 'No', 'No', 'Yes', 'Yes', 'Yes', 'No']
    })

    data_path = os.path.join(tmp_path, "raw.csv")

    sample_features_dataframe.to_csv(data_path, index=False, header=True)

    return data_path

def test_data_ingestion(sample_features, tmp_path, monkeypatch):

    DataIngestion_obj = data_ingestion.DataIngestion()

    monkeypatch.setitem(data_ingestion.config, "data_path", sample_features)
    monkeypatch.setattr(DataIngestion_obj.data_ingestion_config, "train_data_path", tmp_path/"train.csv")
    monkeypatch.setattr(DataIngestion_obj.data_ingestion_config, "test_data_path", tmp_path/"test.csv")
    monkeypatch.setattr(DataIngestion_obj.data_ingestion_config, "raw_data_path", tmp_path/"raw.csv")

    train_data_path, test_data_path = DataIngestion_obj.initiate_data_ingestion()

    train_data = pd.read_csv(train_data_path)
    test_data = pd.read_csv(test_data_path)

    train_data['Date'] = pd.to_datetime(
        train_data['Date']
    )

    assert train_data is not None
    assert test_data is not None
    assert train_data.shape == (8,22)
    assert test_data.shape == (2,22)
    assert list(train_data.columns) == list(test_data.columns) == ['Date', 'Location', 'MinTemp', 'MaxTemp',
                                        'Rainfall', 'Evaporation', 'Sunshine', 'WindGustDir',
                                        'WindGustSpeed', 'WindDir9am', 'WindDir3pm','WindSpeed9am',
                                        'WindSpeed3pm', 'Humidity9am', 'Humidity3pm', 'Pressure9am',
                                        'Pressure3pm','Cloud9am', 'Cloud3pm', 'Temp9am',
                                        'Temp3pm', 'RainToday']
    assert all(pd.api.types.is_numeric_dtype(train_data[col])
               for col in train_data.select_dtypes('number').columns)
    assert all(pd.api.types.is_string_dtype(train_data[col])
               for col in train_data.select_dtypes('object').columns
               if col != 'Date')
    assert pd.api.types.is_datetime64_any_dtype(train_data['Date'])

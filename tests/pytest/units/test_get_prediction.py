import numpy as np
import pandas as pd
import pytest

from src.pipeline.prediction_pipeline import PredictionPipeline


@pytest.fixture
def sample_features():

    sample_features_dataframe = pd.DataFrame(
        {
            'Date': '2012-04-22',
            'Location': 'MountGambier',
            'MinTemp': 15.0,
            'MaxTemp': 18.9,
            'Rainfall': 4.2,
            'Evaporation': 6.6,
            'Sunshine': 8.0,
            'WindGustDir': 'NNW',
            'WindGustSpeed': 54.0,
            'WindDir9am': 'NNW',
            'WindDir3pm': 'NW',
            'WindSpeed9am': 24.0,
            'WindSpeed3pm': 31.0,
            'Humidity9am': 73.0,
            'Humidity3pm': 64.0,
            'Pressure9am': 1005,
            'Pressure3pm': 1003.7,
            'Cloud9am': 4.0,
            'Cloud3pm': 5.0,
            'Temp9am': 17.3,
            'Temp3pm': 17.6,
            'RainToday': 'Yes'
        },
        index=[0]
    )

    return sample_features_dataframe


def test_get_prediction(sample_features):

    PredictionPipeline_obj = PredictionPipeline()

    result = PredictionPipeline_obj.get_prediction(sample_features)

    assert (isinstance(result, dict))
    assert (0 < result['confidence'] < 100)
    assert (result['prediction'] in [0,1])
    assert (isinstance(result['prediction'], (int, np.integer)))
    assert (list(result['features'].columns) == ['MinTemp', 'MaxTemp', 'Rainfall', 'Evaporation', 'Sunshine',
                                                 'WindGustSpeed', 'WindSpeed9am', 'WindSpeed3pm', 'Humidity9am',
                                                 'Humidity3pm', 'Pressure9am', 'Pressure3pm', 'Cloud9am', 'Cloud3pm',
                                                 'Temp9am', 'Temp3pm', 'Year', 'Month', 'Day', 'Weekday', 'Location',
                                                 'WindGustDir', 'WindDir9am', 'WindDir3pm', 'RainToday'])
    assert (all(result['features'][col].dtype == np.float64 for col in result['features'].columns))
    assert (result['features'].shape == (1,25))

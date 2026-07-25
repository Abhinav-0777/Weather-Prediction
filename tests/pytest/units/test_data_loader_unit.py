import os
import pytest
import numpy as np
import pandas as pd
from src.utils import save_object
from src.exception import CustomException
from src.monitoring.data_loader import get_reference_evaluation_dataframe, get_reference_train_dataframe



def test_get_reference_evaluation_dataframe_raises_on_missing_file():

    with pytest.raises(CustomException):
        get_reference_evaluation_dataframe(reference_path="nonexistent_file_that_does_not_exist.csv")



@pytest.fixture
def sample_reference_evaluation_path(tmp_path):

    df = pd.DataFrame({
        'truth_label': [1,0,1,1,0],
        'predicted_output': [1,0,0,0,1]
    })

    file_path = os.path.join(tmp_path, "sample_reference_evaluation.csv")

    df.to_csv(file_path, header=True, index=False)

    return file_path



def test_get_reference_evaluation_dataframe(sample_reference_evaluation_path):

    reference_evaluation_dataframe = get_reference_evaluation_dataframe(reference_path=sample_reference_evaluation_path)

    assert isinstance(reference_evaluation_dataframe, pd.DataFrame)
    assert list(reference_evaluation_dataframe.columns) == ['truth_label','predicted_output']
    assert all(reference_evaluation_dataframe[col].dtype == 'int64' for col in reference_evaluation_dataframe.columns) 
    assert all(set(reference_evaluation_dataframe[col].unique()) == {0,1} for col in reference_evaluation_dataframe.columns)
    assert reference_evaluation_dataframe.shape == (5,2)



def test_get_reference_train_dataframe_raises_on_missing_file():

    with pytest.raises(CustomException):
        get_reference_train_dataframe(reference_path="nonexistent_file_that_does_not_exist.csv")



@pytest.fixture()
def sample_reference_train_path(tmp_path):

    np.random.seed(42)
    sample_reference_train_array = np.random.rand(5,26)

    file_path = os.path.join(tmp_path, "sample_reference_train.npy")

    save_object(file_path, sample_reference_train_array)

    return file_path



def test_get_reference_train_dataframe(sample_reference_train_path):

    reference_train_dataframe = get_reference_train_dataframe(sample_reference_train_path)

    assert isinstance(reference_train_dataframe, pd.DataFrame)
    assert list(reference_train_dataframe.columns) == [
                'MinTemp', 'MaxTemp', 'Rainfall', 'Evaporation', 'Sunshine',
                'WindGustSpeed', 'WindSpeed9am', 'WindSpeed3pm',
                'Humidity9am', 'Humidity3pm', 'Pressure9am', 'Pressure3pm',
                'Cloud9am', 'Cloud3pm', 'Temp9am', 'Temp3pm', 'Year', 'Month',
                'Day', 'Weekday', 'Location', 'WindGustDir', 'WindDir9am',
                'WindDir3pm', 'RainToday', 'truth_label'
            ]
    assert reference_train_dataframe.shape == (5,26)
    assert all(reference_train_dataframe[col].dtype == 'float64' for col in reference_train_dataframe.columns)
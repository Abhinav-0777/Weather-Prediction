import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.components.data_transformation import DataTransformation
from src.exception import CustomException
from src.utils import load_object


@pytest.fixture
def sample_data(tmp_path: Path) :

    train_path = os.path.join(tmp_path, "train.csv")
    test_path = os.path.join(tmp_path, "test.csv")

    data1 = {"Date" : ["2024-02-22", "2024-02-23", "2024-02-24", "2024-02-25"],
        "MaxTemp" : [10.2, 22.1, np.nan, 40.2],
        "WindDir" : ["N", "E", "S", "W"],
        "RainTomorrow" : ["No", "Yes", "Yes", "No"]
    }

    dummy_train_data = pd.DataFrame(data1)


    data2 = {"Date" : ["2024-06-13", "2024-06-14", "2024-06-15", "2024-06-16"],
        "MaxTemp" : [10.9, 19.1, 23.1, 57.2],
        "WindDir" : ["E", np.nan, "W", "W"],
        "RainTomorrow" : ["Yes", "Yes", "No", "Yes"]
    }

    dummy_test_data = pd.DataFrame(data2)

    dummy_train_data.to_csv(train_path, index=False)
    dummy_test_data.to_csv(test_path, index=False)

    return str(train_path), str(test_path)



def test_get_data_transformer_object() :

    numerical_columns = ["MaxTemp"]
    categorical_columns = ["WindDir", "RainTomorrow"]

    data_transformer_object = DataTransformation()

    preprocessor = data_transformer_object.get_data_transformer_object(
        numerical_columns=numerical_columns,
        categorical_columns=categorical_columns
    )

    assert preprocessor is not None
    assert hasattr(preprocessor, "fit")
    assert hasattr(preprocessor, "transform")



@pytest.fixture
def transformed_result(sample_data: tuple[str, str], tmp_path: Path):

    train_path, test_path = sample_data
    data_transformer_object = DataTransformation()

    data_transformer_object.data_transformation_config.transformed_train_array_path = os.path.join(tmp_path, "train_arr.npy")
    data_transformer_object.data_transformation_config.transformed_test_array_path = os.path.join(tmp_path, "test_arr.npy")
    data_transformer_object.data_transformation_config.transformed_data_obj_file_path = os.path.join(tmp_path, "preprocessor.pkl")

    transformed_train_array_path, transformed_test_array_path, preprocessor_path = data_transformer_object.initiate_data_transformation(
            train_data_path=train_path,
            test_data_path=test_path
    )

    train_arr = load_object(transformed_train_array_path)
    test_arr = load_object(transformed_test_array_path)

    return train_arr, test_arr, preprocessor_path



def test_initiate_data_transformation(transformed_result: tuple[np.ndarray, np.ndarray, str]) :

    train_arr, test_arr, _ = transformed_result

    assert train_arr is not None
    assert test_arr is not None
    assert train_arr.shape[0] > 0
    assert test_arr.shape[0] > 0



def test_no_null_values(transformed_result: tuple[np.ndarray, np.ndarray, str]) :

    train_arr, test_arr, _ = transformed_result

    assert not np.isnan(train_arr).any()
    assert not np.isnan(test_arr).any()



def test_shape_consistency(sample_data: tuple[str, str], transformed_result: tuple[np.ndarray, np.ndarray, str]) :

    train_path, test_path = sample_data

    original_train_data = pd.read_csv(train_path)
    original_test_data = pd.read_csv(test_path)

    train_arr, test_arr, _ = transformed_result

    assert train_arr.shape[1] == test_arr.shape[1]
    assert original_train_data.shape[0] == train_arr.shape[0]
    assert original_test_data.shape[0] == test_arr.shape[0]



def test_invalid_input(tmp_path: Path) :

    bad_train_data = pd.DataFrame({
        "MaxTemp": [10, 20],
        "WindDir": ["N", "S"],
        "RainTomorrow": [1, 0]
    })

    bad_test_data = pd.DataFrame({
        "MaxTemp": [15, 25],
        "WindDir": ["W", "E"],
        "RainTomorrow": [1, 1]
    })

    bad_train_path = os.path.join(tmp_path, "bad_train.csv")
    bad_test_path = os.path.join(tmp_path, "bad_test.csv")

    bad_train_data.to_csv(bad_train_path, index=False)
    bad_test_data.to_csv(bad_test_path, index=False)

    data_transformer_object = DataTransformation()

    data_transformer_object.data_transformation_config.transformed_train_array_path = os.path.join(tmp_path, "train_arr.npy")
    data_transformer_object.data_transformation_config.transformed_test_array_path = os.path.join(tmp_path, "test_arr.npy")
    data_transformer_object.data_transformation_config.transformed_data_obj_file_path = os.path.join(tmp_path, "preprocessor.pkl")

    with pytest.raises(CustomException) :
        data_transformer_object.initiate_data_transformation(
            train_data_path=bad_train_path,
            test_data_path=bad_test_path
        )

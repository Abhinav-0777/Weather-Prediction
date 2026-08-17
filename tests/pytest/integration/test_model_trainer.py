import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer
from src.exception import CustomException


@pytest.fixture
def mock_save_object(monkeypatch: pytest.MonkeyPatch) :

    def dummy_save_object_data_transformation(file_path, obj) :
        pass

    monkeypatch.setattr(
        "src.components.data_transformation.save_object",
        dummy_save_object_data_transformation
    )

    def dummy_save_object_model_trainer(file_path, obj) :
        pass

    monkeypatch.setattr(
        "src.components.model_trainer.save_object",
        dummy_save_object_model_trainer
    )



@pytest.fixture
def mock_evaluate_models(monkeypatch: pytest.MonkeyPatch) :

    def dummy_evaluate_models(X_train, y_train, X_test, y_test, models, params) :

        model = models['Random Forest']

        model.fit(X_train, y_train)

        return {'Random Forest':0.8,
                'Logistic Regression':0.7}

    monkeypatch.setattr("src.components.model_trainer.evaluate_models",
                        dummy_evaluate_models)


@pytest.fixture
def prepare_transformed_data(tmp_path: Path, mock_save_object: None):

    train_path = os.path.join(tmp_path, "train.csv")
    test_path = os.path.join(tmp_path, "test.csv")

    train_data = pd.DataFrame({
        "Date": ["2024-02-22", "2024-02-23", "2024-02-24", "2024-02-25"],
        "MaxTemp": [10.2, 22.1, np.nan, 40.2],
        "WindDir": ["N", "E", "S", "W"],
        "RainTomorrow": [0, 1, 1, 0]
    })

    test_data = pd.DataFrame({
        "Date": ["2024-06-13", "2024-06-14", "2024-06-15", "2024-06-16"],
        "MaxTemp": [10.9, 19.1, 23.1, 57.2],
        "WindDir": ["E", np.nan, "W", "W"],
        "RainTomorrow": [1, 1, 0, 1]
    })

    train_data.to_csv(train_path, index=False)
    test_data.to_csv(test_path, index=False)

    transformer = DataTransformation()
    train_arr, test_arr, preprocessor_path = transformer.initiate_data_transformation(
        train_path=str(train_path),
        test_path=str(test_path)
    )

    return train_arr, test_arr, preprocessor_path



def test_initiate_model_trainer(prepare_transformed_data: tuple[np.array, np.array, str], mock_evaluate_models) :

    train_arr, test_arr, preprocessor_path = prepare_transformed_data

    model_trainer_object = ModelTrainer()

    best_model_accuracy = model_trainer_object.initiate_model_trainer(
        train_array=train_arr,
        test_array=test_arr,
        preprocessor_path=preprocessor_path
    )

    assert isinstance(best_model_accuracy, (float, np.floating))
    assert 0 <= best_model_accuracy <= 1



def test_low_accuracy_case(prepare_transformed_data: tuple[np.array, np.array, str], monkeypatch: pytest.MonkeyPatch) :

    def fake_evaluate_models(*args, **kwargs) :
        return {'Logistic Regression': 0.5}

    monkeypatch.setattr('src.components.model_trainer.evaluate_models',
                        fake_evaluate_models)

    train_arr, test_arr, preprocessor_path = prepare_transformed_data

    model_trainer_object = ModelTrainer()

    with pytest.raises(CustomException) :
        model_trainer_object.initiate_model_trainer(
            train_array=train_arr,
            test_array=test_arr,
            preprocessor_path=preprocessor_path
        )

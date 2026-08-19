import os
import sys
from functools import lru_cache

import dill
import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import fbeta_score, make_scorer
from sklearn.model_selection import GridSearchCV

from src.exception import CustomException
from src.logger import logging


def save_object(file_path, obj) :

    try:
        FILE_DIR = os.path.dirname(file_path)

        os.makedirs(FILE_DIR, exist_ok=True)

        with open(file_path, "wb") as file_obj :
            dill.dump(obj, file_obj)


    except Exception as e :
        logging.exception("An error has occurred")
        raise CustomException(e,sys)



def evaluate_models(X_train, y_train, X_test, y_test, models, params) :

    model_report = {}

    try :

        for i in range(len(list(models))) :

            model = list(models.values())[i]
            para = list(params.values())[i]

            f2_scorer = make_scorer(fbeta_score, beta=2)

            gs = GridSearchCV(model, para, cv=5, scoring=f2_scorer, verbose=2, n_jobs= 4 if list(models.keys())[i] in ['XGBoost','CatBoost'] else -1)
            gs.fit(X_train, y_train)

            model.set_params(**gs.best_params_)

            model.fit(X_train, y_train)

            y_test_pred = model.predict(X_test)

            test_model_score = fbeta_score(y_test, y_test_pred, beta=2)

            model_report[list(models.keys())[i]] = test_model_score


        return model_report


    except Exception as e :
        logging.exception("An error has occurred")
        raise CustomException(e,sys)


def load_object(file_path) :

    try :
        with open(file_path,'rb') as file_obj :
            return dill.load(file_obj)

    except Exception as e :
        logging.exception("An error has occurred.")
        raise CustomException(e,sys)


def get_data_Features(train_path) :

    try :
        data = pd.read_csv(train_path)

        features_index = data.columns[:-1]

        features_list = features_index.to_list()

        return features_list

    except Exception as e :
        raise CustomException(e,sys)


@lru_cache(maxsize=1)
def load_config(config_path: str = "config.yaml")-> dict:

    try:
        with open(config_path) as f:
            return yaml.safe_load(f)

    except FileNotFoundError as e:
        logging.exception(f"Config file not found at {config_path}")
        raise CustomException(e,sys)

    except yaml.YAMLError as e:
        logging.exception(f"Error parsing yaml {e}")
        raise CustomException(e,sys)


def make_data_json_serializable(result: dict, metrics: dict) -> dict:

    for key in list(result.keys()):

        value = result[key]
        if isinstance(value, np.integer):
            result[key] = int(value)
            metrics['prediction'] = int(value)
        elif isinstance(value, np.floating):
            result[key] = float(value)
            metrics['confidence_score'] = float(value)
        elif isinstance(value, pd.DataFrame):
            result[key] = value.to_dict(orient="records")
            metrics['input_features'] = value.to_dict(orient="records")

    return result

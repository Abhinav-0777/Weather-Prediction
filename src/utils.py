import os
import sys
import time
from functools import lru_cache

import dill
import mlflow
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
        logging.exception(f"An error has occurred while saving the object to {file_path}")
        raise CustomException(e,sys)



def evaluate_models(X_train, y_train, X_test, y_test, models, params) :

    model_report = {}

    try :

        logging.info(f"Starting hyperparameter tuning and evaluation for {len(models)} models")

        for i in range(len(list(models))) :

            model_name = list(models.keys())[i]
            model = list(models.values())[i]
            para = list(params.values())[i]

            logging.info(f"[{i+1}/{len(models)}] Starting GridSearchCV for model: {model_name}")
            logging.debug(f"Parameter grid for {model_name}: {para}")

            f2_scorer = make_scorer(fbeta_score, beta=2)

            model_start_time = time.time()

            with mlflow.start_run(run_name=model_name, nested=True):

                gs = GridSearchCV(model, para, cv=5, scoring=f2_scorer, verbose=2, n_jobs= 4 if list(models.keys())[i] in ['XGBoost','CatBoost'] else -1)
                gs.fit(X_train, y_train)

                logging.info(f"{model_name}: GridSearchCV completed. Best params: {gs.best_params_}, Best CV F2 score: {gs.best_score_:.4f}")

                model.set_params(**gs.best_params_)

                model.fit(X_train, y_train)

                y_test_pred = model.predict(X_test)

                test_model_score = fbeta_score(y_test, y_test_pred, beta=2)

                model_duration = time.time() - model_start_time

                logging.info(f"{model_name}: Test F2 score: {test_model_score:.4f} (completed in {model_duration:.2f} seconds)")

                mlflow.log_params(gs.best_params_)
                mlflow.log_metric("cv_best_f2_score", gs.best_score_)
                mlflow.log_metric("test_f2_score", test_model_score)
                mlflow.log_param("model_name", model_name)

                logging.debug(f"Logged params and metrics to MLflow for {model_name}")

            model_report[list(models.keys())[i]] = test_model_score

        logging.info(f"Completed evaluation of all {len(models)} models")
        logging.info(f"Final model report: {model_report}")

        return model_report


    except Exception as e :
        logging.exception("An error has occurred while hyperparameter tuning")
        raise CustomException(e,sys)


def load_object(file_path) :

    try :
        with open(file_path,'rb') as file_obj :
            return dill.load(file_obj)

    except Exception as e :
        logging.exception(f"An error has occurred loading the object from {file_path}")
        raise CustomException(e,sys)


def get_data_Features(train_path) :

    try :
        data = pd.read_csv(train_path)

        features_index = data.columns[:-1]

        features_list = features_index.to_list()

        return features_list

    except Exception as e :
        logging.exception(f"An error occurred while fetching the list of features from dataframe at path {train_path}")
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
        logging.exception("Error parsing yaml")
        raise CustomException(e,sys)


def make_data_json_serializable(result: dict, metrics: dict) -> dict:

    try:

        for key in list(result.keys()):

            value = result[key]

            if isinstance(value, np.integer):
                logging.info(f"Converting the value: {value} of key: {key} into python native integer type")
                result[key] = int(value)
                metrics['prediction'] = int(value)

            elif isinstance(value, np.floating):
                logging.info(f"Converting the value: {value} of key: {key} into python native float type")
                result[key] = float(value)
                metrics['confidence_score'] = float(value)

            elif isinstance(value, pd.DataFrame):
                logging.info(f"Converting the value: {value} of key: {key} into list[dict] type")
                result[key] = value.to_dict(orient="records")
                metrics['input_features'] = value.to_dict(orient="records")

        return result

    except Exception as e:
        logging.exception("An error occurred while making 'result' and 'metrics' json serializable")
        raise CustomException(e,sys)

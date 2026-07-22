import os
import sys
import yaml
import dill
import pandas as pd
from functools import lru_cache
from src.logger import logging
from src.exception import CustomException
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import fbeta_score, make_scorer


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
        with open(config_path, "r") as f:
            return yaml.safe_load(f)

    except FileNotFoundError as e:
        logging.exception(f"Config file not found at {config_path}")
        raise CustomException(e,sys)
    
    except yaml.YAMLError as e:
        logging.exception(f"Error parsing yaml {e}")
        raise CustomException(e,sys)
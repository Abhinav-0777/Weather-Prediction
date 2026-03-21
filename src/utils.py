import sys
import os
import pandas as pd
from src.exception import CustomException
from src.logger import logging
import dill
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV

def save_object(file_path, obj) :
    
    try:
        FILE_DIR = os.path.dirname(file_path)

        os.makedirs(FILE_DIR, exist_ok=True)
    
        with open(file_path, "wb") as file_obj :
            dill.dump(obj, file_obj)
    
    
    except Exception as e :
        logging.info("An error has occurred")
        raise CustomException(e,sys)
    


def evaluate_models(X_train, y_train, X_test, y_test, models, params) :

    model_report = {}

    try :

        for i in range(len(list(models))) :

            model = list(models.values())[i]
            para = list(params.values())[i]
 
            gs = GridSearchCV(model, para, cv=5, verbose=2, n_jobs=-1)
            gs.fit(X_train, y_train)

            model.set_params(**gs.best_params_)

            model.fit(X_train, y_train)

            y_train_pred = model.predict(X_train)

            y_test_pred = model.predict(X_test)

            train_model_score = accuracy_score(y_train, y_train_pred)

            test_model_score = accuracy_score(y_test, y_test_pred)

            model_report[list(models.keys())[i]] = test_model_score


        return model_report

    
    except Exception as e :
        logging.error("An error has occurred")
        raise CustomException(e,sys)


def load_object(file_path) :

    try :
        with open(file_path,'rb') as file_obj :
            return dill.load(file_obj)

    except Exception as e :
        raise CustomException(e,sys)


def get_data_Features(train_path) :

    try :
        data = pd.read_csv(train_path) 

        features_index = data.columns[:-1]

        features_list = features_index.to_list()

        return features_list

    except Exception as e :
        raise CustomException(e,sys)
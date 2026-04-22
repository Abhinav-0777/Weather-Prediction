import os
import sys
import shap
import numpy as np
import pandas as pd
from src.utils import load_object
from src.logger import logging
from src.exception import CustomException


def model_interpretability(features: pd.DataFrame) -> list:
    
    try :

        model_path = os.path.join("artifacts","model.pkl")

        logging.info("Loading the trained model object")

        model_obj = load_object(model_path)

        logging.info("Making the explainer object")

        explainer = shap.TreeExplainer(model_obj)
        
        logging.info("Getting SHAP values for the data")

        shap_values = explainer(features)

        logging.info("Sorting the list")

        values = shap_values.values[0]
        values = np.linalg.norm(values, axis=1)
        
        sorted_list = sorted(list(zip(list(features.columns), values)), key=lambda x: abs(x[1]), reverse=True)
        
        top_features = [
                    {"name": name, "value": value}
                    for name, value in sorted_list[:3]
                ]

        return top_features

    except Exception as e:
        logging.error("An error has occurred.")
        raise CustomException(e,sys)
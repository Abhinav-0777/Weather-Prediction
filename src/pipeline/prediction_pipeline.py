import pandas as pd
import sys
import os
from src.exception import CustomException
from src.logger import logging
from src.utils import load_object

class PredictionPipeline :

    def __init__(self) :
        pass 

    def get_prediction(self, features) :
        
        """This function gets the user data as arguement and runs prediction on it .

        Returns:
            list : the prediction of the model on that particular data .
        """

        try :
            model_path = os.path.join("artifacts","model.pkl")
            preprocessor_path = os.path.join("artifacts","preprocessing_object.pkl")
            
            logging.info("loading the model path and preprocessor object")

            model_obj = load_object(model_path)
            preprocessor_object = load_object(preprocessor_path)

            logging.info("Applying the data transformation on the user entered data")

            features['Date'] = pd.to_datetime(features['Date'])

            features['Year'] = features['Date'].dt.year
            features['Month'] = features['Date'].dt.month
            features['Day'] = features['Date'].dt.day
            features['Weekday'] = features['Date'].dt.weekday
                    
            features = features.drop(columns = 'Date')

            data_scaled = preprocessor_object.transform(features)

            logging.info("Running prediction on the scaled data")

            preds = model_obj.predict(data_scaled)

            return preds
        

        except Exception as e :
            raise CustomException(e,sys)
        

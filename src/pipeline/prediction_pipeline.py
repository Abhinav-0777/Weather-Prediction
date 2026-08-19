import sys

import pandas as pd

from src.exception import CustomException
from src.logger import logging
from src.utils import load_config, load_object

config = load_config()

class PredictionPipeline :

    def __init__(self) :
        pass

    def get_prediction(self, features) -> dict:

        """This function gets the user data as arguement and runs prediction on it .

        Returns:

            dict : with these following keys

                prediction : the prediction of the model on that particular data
                features : the scaled dataframe of that particular data
                confidence : the confidence value of the prediction
        """

        try :

            model_obj = load_object(config['model_path'])
            preprocessor_object = load_object(config['preprocessor_path'])

            logging.info("Applying the data transformation on the user entered data")

            features['Date'] = pd.to_datetime(features['Date'])

            features['Year'] = features['Date'].dt.year
            features['Month'] = features['Date'].dt.month
            features['Day'] = features['Date'].dt.day
            features['Weekday'] = features['Date'].dt.weekday

            features = features.drop(columns = 'Date')

            data_scaled = preprocessor_object.transform(features)

            data_scaled_dataframe = pd.DataFrame(data_scaled, columns=config['column_names'])

            logging.info("Running prediction on the scaled data")

            preds = model_obj.predict(data_scaled)

            proba = model_obj.predict_proba(data_scaled)

            confidence = proba[0][preds[0]]

            return {
                    "prediction": preds[0],
                    "features": data_scaled_dataframe,
                    "confidence": confidence
                }


        except Exception as e :
            logging.exception("An error has occurred while getting the prediction from the model")
            raise CustomException(e,sys)


if __name__ == "__main__":

    logging.info("Running prediction_pipeline.py as a standalone script")

    test_data = pd.read_csv(config['test_path'])
    logging.info(f"Loaded test data with {test_data.shape[0]} rows and {test_data.shape[1]} columns")

    row_0 = test_data.iloc[[0]]
    logging.info("Extracted first row from test data for prediction")

    PredictionPipeline_obj = PredictionPipeline()
    result = PredictionPipeline_obj.get_prediction(row_0)

    logging.info(f"Prediction: {result['prediction']}, Confidence: {result['confidence']}")

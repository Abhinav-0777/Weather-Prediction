import pandas as pd
import sys
from fastapi import FastAPI
from src.schema import data_validation
from src.pipeline.prediction_pipeline import PredictionPipeline
from src.exception import CustomException

app = FastAPI()

@app.get('/health')
def health() :
    return {'Health':'Ok, API is running.'}


@app.post('/predict')
def prediction(data : data_validation) :
   
    try :
        features_dict = data.model_dump()

        features_dataframe = pd.DataFrame([features_dict])

        PredictionPipeline_obj = PredictionPipeline()

        final_pred = PredictionPipeline_obj.get_prediction(features_dataframe)

        return {"Prediction" : final_pred.tolist()}

    except Exception as e :
        raise CustomException(e,sys)
    
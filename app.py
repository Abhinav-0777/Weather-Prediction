import pandas as pd
import sys
from src.logger import logging
from fastapi import FastAPI, Request, Form
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from src.schema import data_validation
from src.pipeline.prediction_pipeline import PredictionPipeline
from src.exception import CustomException

app = FastAPI()

logging.info("mounting the static folder to our web app")

app.mount("/static", StaticFiles(directory='static'), name="static")

logging.info("creating Jinja2Templates object")

templates = Jinja2Templates(directory="templates")

@app.get('/health')
def health() :
    return {'Health':'Ok, API is running.'}

@app.get("/", response_class=HTMLResponse)
def home(request : Request) :
    return templates.TemplateResponse("home.html", {"request":request})
    

@app.post('/predict', response_class=HTMLResponse)
def prediction( 
    request: Request,

    Date: str = Form(...),
    Location: str = Form(...),
    MinTemp: float = Form(...),
    MaxTemp: float = Form(...),
    Rainfall: float = Form(...),
    Evaporation: float = Form(...),
    Sunshine: float = Form(...),
    WindGustDir: str = Form(...),
    WindGustSpeed: float = Form(...),
    WindDir9am: str = Form(...),
    WindDir3pm: str = Form(...),
    WindSpeed9am: float = Form(...),
    WindSpeed3pm: float = Form(...),
    Humidity9am: float = Form(...),
    Humidity3pm: float = Form(...),
    Pressure9am: float = Form(...),
    Pressure3pm: float = Form(...),
    Cloud9am: float = Form(...),
    Cloud3pm: float = Form(...),
    Temp9am: float = Form(...),
    Temp3pm: float = Form(...),
    RainToday: str = Form(...)
):

    try :

        logging.info("validating data using pydantic models")

        data = data_validation(
            Date=Date,
            Location=Location,
            MinTemp=MinTemp,
            MaxTemp=MaxTemp,
            Rainfall=Rainfall,
            Evaporation=Evaporation,
            Sunshine=Sunshine,
            WindGustDir=WindGustDir,
            WindGustSpeed=WindGustSpeed,
            WindDir9am=WindDir9am,
            WindDir3pm=WindDir3pm,
            WindSpeed9am=WindSpeed9am,
            WindSpeed3pm=WindSpeed3pm,
            Humidity9am=Humidity9am,
            Humidity3pm=Humidity3pm,
            Pressure9am=Pressure9am,
            Pressure3pm=Pressure3pm,
            Cloud9am=Cloud9am,
            Cloud3pm=Cloud3pm,
            Temp9am=Temp9am,
            Temp3pm=Temp3pm,
            RainToday=RainToday
        )

        logging.info("converting user entered data to a dataframe")

        df = pd.DataFrame([data.model_dump()])

        logging.info("getting the predition from the model")

        pipeline = PredictionPipeline()
        pred = pipeline.get_prediction(df)

        return templates.TemplateResponse(
            "home.html",
            {
                "request": request,
                "prediction": f"Prediction: {pred[0]}"
            }
        )

    except Exception as e :
        logging.error("an error has occurred")
        raise CustomException(e,sys)
    
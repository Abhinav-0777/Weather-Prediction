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
from src.components.data_ingestion import fetch_weather
from datetime import datetime


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
    


@app.get('/predict_live', response_class=HTMLResponse)
def prediction_live(request: Request) :
    
    try :
        longitude, latitude = 144.9633, -37.814
        hourly_features = ["temperature_2m","relative_humidity_2m","pressure_msl","wind_speed_10m","wind_direction_10m","wind_gusts_10m","cloud_cover"]
        daily_features = ["temperature_2m_max","temperature_2m_min","sunshine_duration","precipitation_sum","et0_fao_evapotranspiration","wind_direction_10m_dominant"]

        pred = fetch_weather(
            latitude=latitude,
            longitude=longitude,
            hourly_features=hourly_features,
            daily_features=daily_features
        )

        directions = ['N','NNE','NE','ENE','E','ESE','SE','SSE','S','SSW','SW','WSW','W','WNW','NW','NNW']

        data = data_validation(
            Date=datetime.now().strftime("%Y-%m-%d"),
            Location="Melbourne",
            MinTemp=pred["daily"]["temperature_2m_min"][0],
            MaxTemp=pred["daily"]["temperature_2m_max"][0],
            Rainfall=pred["daily"]["precipitation_sum"][0],
            Evaporation=pred["daily"]["et0_fao_evapotranspiration"][0],
            Sunshine=pred["daily"]["sunshine_duration"][0]/3600,
            WindGustDir=directions[int((pred["daily"]["wind_direction_10m_dominant"][0] + 11.25) / 22.5) % 16],
            WindGustSpeed=max(pred["hourly"]["wind_gusts_10m"]),
            WindDir9am=directions[int((pred["hourly"]["wind_direction_10m"][9] + 11.25) / 22.5) % 16],
            WindDir3pm=directions[int((pred["hourly"]["wind_direction_10m"][15] + 11.25) / 22.5) % 16],
            WindSpeed9am=pred["hourly"]["wind_speed_10m"][9],
            WindSpeed3pm=pred["hourly"]["wind_speed_10m"][15],
            Humidity9am=pred["hourly"]["relative_humidity_2m"][9],
            Humidity3pm=pred["hourly"]["relative_humidity_2m"][15],
            Pressure9am=pred["hourly"]["pressure_msl"][9],
            Pressure3pm=pred["hourly"]["pressure_msl"][15],
            Cloud9am=pred["hourly"]["cloud_cover"][9]/10,
            Cloud3pm=pred["hourly"]["cloud_cover"][15]/10,
            Temp9am=pred["hourly"]["temperature_2m"][9],
            Temp3pm=pred["hourly"]["temperature_2m"][15],
            RainToday="Yes" if pred["daily"]["precipitation_sum"][0]>0 else "No"
        )

        df = pd.DataFrame([data.model_dump()])

        pipeline = PredictionPipeline()
        pred = pipeline.get_prediction(df)

        return templates.TemplateResponse(
                "home.html",
                {
                    "request": request,
                    "prediction": f"Prediction: {pred[0]}"
                }
            )

    except Exception as e:
        logging.error("An error has occurred.")
        raise CustomException(e,sys)

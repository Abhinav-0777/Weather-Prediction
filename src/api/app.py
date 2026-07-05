import sys
import pandas as pd
from src.logger import logging
from slowapi import Limiter
from slowapi.util import get_remote_address
from fastapi import FastAPI, Request, Form
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from src.schema import data_validation
from src.pipeline.prediction_pipeline import PredictionPipeline
from src.exception import CustomException
from src.components.data_ingestion import fetch_weather
from src.services.explainability_service import model_interpretability
from datetime import datetime


app = FastAPI()

logging.info("mounting the static folder to our web app")

app.mount("/static", StaticFiles(directory='src/api/static'), name="static")

logging.info("creating Jinja2Templates object")

templates = Jinja2Templates(directory="src/api/templates")

@app.get('/health')
def health() :
    return {'Health':'Ok, API is running.'}

@app.on_event("startup")
def startup_event() :
    global app_start_time
    app_start_time = datetime.now() 

def format_uptime(delta):
    total_seconds = int(delta.total_seconds())
    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    seconds = total_seconds % 60
    return f"{hours}h {minutes}m {seconds}s"

@app.get("/info")
def info() :
    return {
        "uptime" : format_uptime(datetime.now() - app_start_time) if app_start_time is not None else "app didn't started properly",
        "model" : "Weather-Prediction Model",
        "version" : "v1"
    }

@app.get("/", response_class=HTMLResponse)
def home(request : Request) :
    return templates.TemplateResponse("home.html", {"request":request})
    


limiter = Limiter(key_func=get_remote_address)

@app.post('/predict', response_class=HTMLResponse)
@limiter.limit("5/minute")
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
        result = pipeline.get_prediction(df)  

        logging.info("Successfully predicted the data")

        logging.info("Getting the model interpreted")

        top_features = model_interpretability(result["features"])

        return templates.TemplateResponse(
            "home.html",
            {
                "request": request,
                "prediction": f"Prediction: {result['prediction']} ({result['confidence']*100:.2f}%)",
                "top_features": top_features
            }
        )

    except Exception as e :
        logging.error("an error has occurred")
        raise CustomException(e,sys)
    


@app.get('/predict_live', response_class=HTMLResponse)
@limiter.limit("5/minute")
def prediction_live(request: Request, location:str = None) :
    
    try :

        CITY_COORDS = {
            "Albury":         (-36.0748, 146.924),
            "BadgerysCreek":  (-33.8907, 150.7426),
            "Cobar":          (-31.4967, 145.8344),
            "CoffsHarbour":   (-30.2963, 153.1135),
            "Moree":          (-29.4628, 149.8416),
            "Newcastle":      (-32.9295, 151.7801),
            "NorahHead":      (-33.2732, 151.5588),
            "NorfolkIsland":  (-29.0408, 167.9547),
            "Penrith":        (-33.75, 150.7),
            "Richmond":       (-41.3333, 173.1833),
            "Sydney":         (-33.8678, 151.2073),
            "SydneyAirport":  (-33.9461, 151.1770),
            "WaggaWagga":     (-35.1258, 147.3537),
            "Williamtown":    (-32.8064, 151.8436),
            "Wollongong":     (-34.424, 150.8935),
            "Canberra":       (-35.2835, 149.1281),
            "Tuggeranong":    (-35.4165, 149.0695),
            "MountGinini":    (-35.5307, 148.7713),
            "Ballarat":       (-37.5662, 143.8496),
            "Bendigo":        (-36.7582, 144.2802),
            "Sale":           (-38.111, 147.068),
            "MelbourneAirport": (-37.6707, 144.8379),
            "Melbourne":      (-37.814, 144.9633),
            "Mildura":        (-34.1855, 142.1625),
            "Nhil":           (-36.3333, 141.65),
            "Portland":       (-38.3456, 141.6042),
            "Watsonia":       (-37.7167, 145.0833),
            "Dartmoor":       (-37.9222, 141.2749),
            "Brisbane":       (-27.4679, 153.0281),
            "Cairns":         (-16.9237, 145.7661),
            "GoldCoast":      (-28.0003, 153.4309),
            "Townsville":     (-19.2664, 146.8057),
            "Adelaide":       (-34.9287, 138.5986),
            "MountGambier":   (-37.8318, 140.7792),
            "Nuriootpa":      (-34.4682, 138.9977),
            "Woomera":        (-31.1998, 136.8326),
            "Albany":         (-35.0269, 117.8837),
            "Witchcliffe":    (-34.0333, 115.1),
            "PearceRAAF":     (-31.6667, 116.0167),
            "PerthAirport":   (-31.9321, 115.9564),
            "Perth":          (-31.9522, 115.8614),
            "SalmonGums":     (-32.9833, 121.6333),
            "Walpole":        (-34.976, 116.7302),
            "Hobart":         (-42.8794, 147.3294),
            "Launceston":     (-41.4388, 147.1347),
            "AliceSprings":   (-23.6975, 133.8836),
            "Darwin":         (-12.4611, 130.8418),
            "Katherine":      (-14.4652, 132.2635),
            "Uluru":          (-25.3415, 131.0354),
        }

        latitude, longitude = CITY_COORDS.get(location,CITY_COORDS['Melbourne'])
        hourly_features = ["temperature_2m","relative_humidity_2m","pressure_msl","wind_speed_10m","wind_direction_10m","wind_gusts_10m","cloud_cover"]
        daily_features = ["temperature_2m_max","temperature_2m_min","sunshine_duration","precipitation_sum","et0_fao_evapotranspiration","wind_direction_10m_dominant"]

        logging.info("fetching live data")

        live_data = fetch_weather(
            latitude=latitude,
            longitude=longitude,
            hourly_features=hourly_features,
            daily_features=daily_features
        )

        directions = ['N','NNE','NE','ENE','E','ESE','SE','SSE','S','SSW','SW','WSW','W','WNW','NW','NNW']

        logging.info("Validating the data from pydantic")

        data = data_validation(
            Date=datetime.now().strftime("%Y-%m-%d"),
            Location="Melbourne",
            MinTemp=live_data["daily"]["temperature_2m_min"][0],
            MaxTemp=live_data["daily"]["temperature_2m_max"][0],
            Rainfall=live_data["daily"]["precipitation_sum"][0],
            Evaporation=live_data["daily"]["et0_fao_evapotranspiration"][0],
            Sunshine=live_data["daily"]["sunshine_duration"][0]/3600,
            WindGustDir=directions[int((live_data["daily"]["wind_direction_10m_dominant"][0] + 11.25) / 22.5) % 16],
            WindGustSpeed=max(live_data["hourly"]["wind_gusts_10m"]),
            WindDir9am=directions[int((live_data["hourly"]["wind_direction_10m"][9] + 11.25) / 22.5) % 16],
            WindDir3pm=directions[int((live_data["hourly"]["wind_direction_10m"][15] + 11.25) / 22.5) % 16],
            WindSpeed9am=live_data["hourly"]["wind_speed_10m"][9],
            WindSpeed3pm=live_data["hourly"]["wind_speed_10m"][15],
            Humidity9am=live_data["hourly"]["relative_humidity_2m"][9],
            Humidity3pm=live_data["hourly"]["relative_humidity_2m"][15],
            Pressure9am=live_data["hourly"]["pressure_msl"][9],
            Pressure3pm=live_data["hourly"]["pressure_msl"][15],
            Cloud9am=live_data["hourly"]["cloud_cover"][9]/10,
            Cloud3pm=live_data["hourly"]["cloud_cover"][15]/10,
            Temp9am=live_data["hourly"]["temperature_2m"][9],
            Temp3pm=live_data["hourly"]["temperature_2m"][15],
            RainToday="Yes" if live_data["daily"]["precipitation_sum"][0]>0 else "No"
        )

        logging.info("Converted to dataframe")

        df = pd.DataFrame([data.model_dump()])

        logging.info("Getting predictions from pipeline")

        pipeline = PredictionPipeline()
        result = pipeline.get_prediction(df)

        logging.info("Successfully predicted the data")

        logging.info("Getting the model interpreted")

        top_features = model_interpretability(result["features"])

        return templates.TemplateResponse(
                "home.html",
                {
                    "request": request,
                    "prediction": f"Prediction: {result['prediction']} ({result['confidence']*100:.2f}%)",
                    "top_features": top_features
                }
            )

    except Exception as e:
        logging.error("An error has occurred.")
        raise CustomException(e,sys)

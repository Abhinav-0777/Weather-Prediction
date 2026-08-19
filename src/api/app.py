import asyncio
import sys
from datetime import datetime

import pandas as pd
from fastapi import BackgroundTasks, Depends, FastAPI, Form, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from prometheus_client import make_asgi_app
from slowapi import Limiter
from slowapi.util import get_remote_address

from src.api.auth import verify_api_key
from src.exception import CustomException
from src.logger import logging
from src.pipeline.prediction_pipeline import PredictionPipeline
from src.schema import data_validation
from src.services.database import save_to_database
from src.services.explainability_service import model_interpretability
from src.services.prediction_service import run_prediction
from src.utils import load_config, make_data_json_serializable

config = load_config()

app = FastAPI()

prometheus_metrics_app = make_asgi_app()
app.mount('/metrics', prometheus_metrics_app)

logging.info("Prometheus /metrics endpoint mounted successfully")

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
        logging.exception("an error has occurred")
        raise CustomException(e,sys)



@app.get('/predict_live', response_class=HTMLResponse)
@limiter.limit('5/minute')
async def prediction_live(request: Request, location: str = None, background_tasks: BackgroundTasks = BackgroundTasks()):

    try:

        output = await run_prediction(location=location, client_type="common_user", model_version=config['model_version'])
        result, metrics = output["result"], output["metrics"]

        logging.info("Getting the model interpreted")
        top_features = await asyncio.to_thread(model_interpretability, result["features"])

        logging.info("Making data json serializable")
        result = make_data_json_serializable(result)

        logging.info("Saving the predictions data and other metrics to Supabase")
        background_tasks.add_task(save_to_database, metrics)

        return templates.TemplateResponse(
            "home.html",
            {
                "request": request,
                "prediction": f"Prediction: {result['prediction']} ({result['confidence']*100:.2f}%)",
                "top_features": top_features
            }
        )

    except Exception as e:
        logging.exception("An error has occurred.")
        raise CustomException(e, sys)



@app.get('/api/predict_live')
@limiter.limit('100/minute')
async def prediction_live_with_api(request: Request, location: str | None = None, background_tasks: BackgroundTasks = BackgroundTasks(), dependencies=Depends(verify_api_key)) -> dict:

    try:

        output = await run_prediction(location=location, client_type="authorized_client", model_version=config['model_version'])
        result, metrics = output["result"], output["metrics"]

        logging.info("Making data json serializable")
        result = make_data_json_serializable(result)

        logging.info("Saving the predictions data and other metrics to Supabase")
        background_tasks.add_task(save_to_database, metrics)

        result['request_id'] = metrics['request_id']

        return result

    except Exception as e:
        logging.exception("An error has occurred.")
        raise CustomException(e, sys)

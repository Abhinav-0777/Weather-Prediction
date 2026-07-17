import sys
import time
import requests
from src.logger import logging
from src.config import get_env
from datetime import datetime, timezone
from src.exception import CustomException


config = get_env()

APP_URL = config.get("APP_URL")
HEALTH_URL = config.get("HEALTH_URL")
API_KEY = config.get("API_KEY")


def check_health()-> dict:
    
    """This function is used to verify that the API is running and responsive.

    Returns:
        dict: A simple status message confirming the API is live.
    """

    try:
        response = requests.get(
            url=HEALTH_URL,
            timeout= 360,
        )
        if response.status_code == 200:
            logging.info("The server is awake and running.")
            return {'Health': 'Ok, The API is running.'}

    except requests.exceptions.RequestException as e:
        logging.exception("Waking up the service failed due to {e}")
        raise CustomException(e,sys)
    
check_health()



CITIES = [
    'Albury', 'BadgerysCreek', 'Cobar', 'CoffsHarbour', 'Moree',
    'Newcastle', 'NorahHead', 'NorfolkIsland', 'Penrith', 'Richmond',
    'Sydney', 'SydneyAirport', 'WaggaWagga', 'Williamtown',
    'Wollongong', 'Canberra', 'Tuggeranong', 'MountGinini', 'Ballarat',
    'Bendigo', 'Sale', 'MelbourneAirport', 'Melbourne', 'Mildura',
    'Nhil', 'Portland', 'Watsonia', 'Dartmoor', 'Brisbane', 'Cairns',
    'GoldCoast', 'Townsville', 'Adelaide', 'MountGambier', 'Nuriootpa',
    'Woomera', 'Albany', 'Witchcliffe', 'PearceRAAF', 'PerthAirport',
    'Perth', 'SalmonGums', 'Walpole', 'Hobart', 'Launceston',
    'AliceSprings', 'Darwin', 'Katherine', 'Uluru'
]

success_count = 0
failed_cities = []

for city in CITIES:

    try:

        logging.info("Sending the request to /api/predict_live endpoint.")

        response = requests.get(
            url= APP_URL,
            headers={"X-API-Key": API_KEY},
            params={"location": city},
            timeout=20
        )
        response.raise_for_status()
        print(f"[{datetime.now(timezone.utc).isoformat()}] {city}: {response.json()}")
        success_count += 1

    except Exception as e:
        print(f"[{datetime.now(timezone.utc).isoformat()}] FAILED for {city}: {e}")
        failed_cities.append(city)
        logging.exception(f"Error {e} has occurred.")

    time.sleep(1)  

print(f"\nDone. Success: {success_count}/{len(CITIES)}")
if failed_cities:
    print(f"Failed cities: {failed_cities}")

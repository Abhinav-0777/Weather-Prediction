import os
import time
import requests
from dotenv import load_dotenv
from src.logger import logging
from datetime import datetime, timezone


load_dotenv()

APP_URL = os.getenv("APP_URL")
API_KEY_2 = os.getenv("API_KEY")

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
            APP_URL,
            headers={"X-API-Key": API_KEY_2},
            params={"location": city},
            timeout=20
        )
        response.raise_for_status()
        print(f"[{datetime.now(timezone.utc).isoformat()}] {city}: {response.json()}")
        success_count += 1

    except Exception as e:
        print(f"[{datetime.now(timezone.utc).isoformat()}] FAILED for {city}: {e}")
        failed_cities.append(city)
        logging.error(f"Error {e} has occurred.")

    time.sleep(1)  

print(f"\nDone. Success: {success_count}/{len(CITIES)}")
if failed_cities:
    print(f"Failed cities: {failed_cities}")

import random

from locust import HttpUser, constant, task


class MOCK_USER(HttpUser):

    wait_time = constant(0)

    @task
    def mock_user_journey(self):

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

        random_city = random.choice(CITIES)

        payload = {
            'Location': random_city
        }

        self.client.get('/predict_live', params=payload)

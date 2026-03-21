from pydantic import BaseModel,Field
from datetime import date
from typing import Literal

class data_validation(BaseModel) :

    Date : date
    Location : Literal['Albury','BadgerysCreek','Cobar','CoffsHarbour','Moree','Newcastle',
                        'NorahHead','NorfolkIsland','Penrith','Richmond','Sydney','SydneyAirport',
                        'WaggaWagga','Williamtown','Wollongong','Canberra','Tuggeranong',
                        'MountGinini','Ballarat','Bendigo','Sale','MelbourneAirport','Melbourne',
                        'Mildura','Nhil','Portland','Watsonia','Dartmoor','Brisbane','Cairns',
                        'GoldCoast','Townsville','Adelaide','MountGambier','Nuriootpa','Woomera',
                        'Albany','Witchcliffe','PearceRAAF','PerthAirport','Perth','SalmonGums',
                        'Walpole','Hobart','Launceston','AliceSprings','Darwin','Katherine',
                        'Uluru'] 
    MinTemp : int = Field (ge=-15, le=45)
    MaxTemp : int = Field (ge=-10, le=52)
    Rainfall : int = Field (ge=0, le=400)
    Evaporation : int = Field (ge=0, le=160)
    Sunshine : int = Field(ge=0, le=15)
    WindGustDir : Literal['W','WNW','WSW','NE','NNW','N','NNE','SW','ENE','SSE','S','NW','SE',
                          'ESE','E','SSW']
    WindGustSpeed : int = Field(ge=0, le=180)
    WindDir9am : Literal['W','NNW','SE','ENE','SW','SSE','S','NE','SSW','N','WSW','ESE','E','NW',
                         'WNW','NNE']
    WindDir3pm : Literal['WNW','WSW','E','NW','W','SSE','ESE','ENE','NNW','SSW','SW','SE','N','S',
                         'NNE','NE']
    WindSpeed9am : int = Field(ge=0, le=150)
    WindSpeed3pm : int = Field(ge=0, le=100)
    Humidity9am : int = Field(ge=0, le=100)
    Humidity3pm : int = Field(ge=0, le=100)
    Pressure9am : int = Field(ge=900, le=1100)
    Pressure3pm : int = Field(ge=900, le=1100)
    Cloud9am : int = Field(ge=0, le=10)
    Cloud3pm : int = Field(ge=0, le=10)
    Temp9am : int = Field(ge=-12, le=45)
    Temp3pm : int = Field(ge=-12, le=52)
    RainToday : Literal['Yes', 'No']
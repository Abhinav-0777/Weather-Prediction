import os
import sys
import httpx
import asyncio
import pandas as pd
from datetime import datetime
from src.logger import logging
from dataclasses import dataclass
from src.exception import CustomException
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer
from src.components.model_evaluation import model_evaluation
from sklearn.model_selection import train_test_split


@dataclass
class DataIngestionConfig :
    train_data_path : str = os.path.join("artifacts","train.csv")
    test_data_path : str = os.path.join("artifacts","test.csv")
    raw_data_path : str = os.path.join("artifacts","raw.csv")

class DataIngestion :

    def __init__(self) :
        self.data_ingestion_config = DataIngestionConfig()
    

    def initiate_data_ingestion(self) :

        """
        Initiates the data ingestion process.

        This function reads the raw dataset from the source, splits the data into
        training and testing datasets, and stores them in the artifacts directory
        for further processing in the ML pipeline.

        Steps performed:
        1. Read the dataset from the specified data source.
        2. Split the dataset into train and test sets.
        3. Save the train and test datasets as CSV files in the artifacts folder.

        Returns:
            tuple: File paths of the saved training and testing datasets.
                (train_data_path, test_data_path)

        Raises:
            CustomException: If any error occurs during the ingestion process.
        """

        logging.info("Data ingestion has started ")

        try :

            df = pd.read_csv("data/weatherAUS.csv")
            logging.info("Successfully read the dataset as a dataframe")

            os.makedirs(os.path.dirname(self.data_ingestion_config.train_data_path),exist_ok=True)
            
            df.to_csv(self.data_ingestion_config.raw_data_path,index=False, header=True)
            
            logging.info("Train test split initiated")

            training_set, testing_set = train_test_split(df, test_size=0.2, random_state=42)
            
            train_set = pd.DataFrame(training_set)
            test_set = pd.DataFrame(testing_set)

            logging.info("Storing the train and test data.....")
 
            train_set.to_csv(self.data_ingestion_config.train_data_path, index=False, header=True)
            test_set.to_csv(self.data_ingestion_config.test_data_path, index=False, header=True)

            logging.info("Data ingestion completed successfully")

            return (self.data_ingestion_config.train_data_path, 
                    self.data_ingestion_config.test_data_path)
        

        except Exception as e :
            logging.exception("An error has occurred")
            raise CustomException(e,sys)
    

logging.info("Creating a client using httpx")
client = httpx.AsyncClient()


async def fetch_weather(latitude:float, longitude:float, hourly_features:list[str], daily_features:list[str], retries:int = 5) -> dict :

    url = "https://api.open-meteo.com/v1/forecast"

    params = {
        "latitude": latitude,
        "longitude": longitude,
        "daily": daily_features,
        "hourly": hourly_features,
        "timezone":"auto",
        "start_date": datetime.now().strftime("%Y-%m-%d"),
        "end_date": datetime.now().strftime("%Y-%m-%d")
    }

    for attempt in range(retries):
            
        try:

            logging.info("Sending the request to the Open-Meteo Weather API")

            response = await client.get(
                url,
                params=params,
                timeout=10
            )
            response.raise_for_status()

            logging.info("Successfully got the data, now converting it to json")

            data = response.json()

            return data 

        except Exception as e:

            if attempt == retries-1:
                logging.exception(f"Due to {e} reason the {attempt}th and last attempt has been failed.")
                raise CustomException(e,sys)
            
            wait_time = 3 * (attempt+1)
            logging.exception(f"Attempt {attempt+1}/{retries} failed: {e}. Retrying in {wait_time}s...")
        
            await asyncio.sleep(wait_time)


if __name__ == "__main__" :
    logging.info("Logging has started")
    
    data_ingestion_obj = DataIngestion()
    train_data, test_data = data_ingestion_obj.initiate_data_ingestion()

    data_transformation_obj = DataTransformation()
    train_transformed, test_transformed, preprocessor_obj = data_transformation_obj.initiate_data_transformation(train_path=train_data, test_path=test_data)

    model_trainer_obj = ModelTrainer()
    best_model_f2_score = model_trainer_obj.initiate_model_trainer(
        train_array=train_transformed,
        test_array=test_transformed,
        preprocessor_path=preprocessor_obj
    )
    metrics_dict = model_evaluation()

    print(best_model_f2_score)
    print(metrics_dict)
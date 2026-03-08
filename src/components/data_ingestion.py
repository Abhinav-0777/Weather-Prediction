import os
from src.logger import logging
from src.exception import CustomException
from src.components.data_transformation import DataTransformation
from dataclasses import dataclass
import pandas as pd
import sys
from sklearn.model_selection import train_test_split

@dataclass
class DataIngestionConfig :
    train_data_path : str = os.path.join("artifacts","train.csv")
    test_data_path : str = os.path.join("artifacts","test.csv")
    raw_data_path : str = os.path.join("artifacts","raw.csv")

class DataIngestion :

    def __init__(self) :
        self.DataIngestionConfig = DataIngestionConfig()
    

    def initiate_data_ingestion(self) :

        logging.info("Data ingestion started by reading a .csv file")

        try :

            df = pd.read_csv(r"data\weatherAUS.csv")
            logging.info("Read the dataset as a dataframe")

            os.makedirs(os.path.dirname(self.DataIngestionConfig.train_data_path),exist_ok=True)
            
            df.to_csv(self.DataIngestionConfig.raw_data_path,index=False, header=True)
            
            logging.info("Train test split initiated")

            training_set, testing_set = train_test_split(df, test_size=0.2, random_state=42)
            
            train_set = pd.DataFrame(training_set)
            test_set = pd.DataFrame(testing_set)

            train_set.to_csv(self.DataIngestionConfig.train_data_path, index=False, header=True)
            test_set.to_csv(self.DataIngestionConfig.test_data_path, index=False, header=True)

            logging.info("Data ingestion completed successfully")

            return (self.DataIngestionConfig.train_data_path, 
                    self.DataIngestionConfig.test_data_path)
        

        except Exception as e :
            logging.error("An error has occurred")
            raise CustomException(e,sys)
    
    

if __name__ == "__main__" :
    logging.info("Logging has started")
    
    data_ingestion_obj = DataIngestion()
    train_data, test_data = data_ingestion_obj.initiate_data_ingestion()

    data_transformation_obj = DataTransformation()
    data_transformation_obj.initiate_data_transformation(train_path=train_data, test_path=test_data)

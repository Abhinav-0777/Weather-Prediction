import sys
import os
from src.exception import CustomException
from src.logger import logging
import dill

def save_object(file_path, obj) :
    
    try:
        FILE_DIR = os.path.dirname(file_path)

        os.makedirs(FILE_DIR, exist_ok=True)
    
        with open(file_path, "wb") as file_obj :
            dill.dump(obj, file_obj)
    
    
    except Exception as e :
        logging.info("An error has occurred")
        raise CustomException(e,sys)
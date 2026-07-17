import sys
from src.config import get_env
from src.logger import logging
from fastapi import Security, HTTPException, status
from fastapi.security import APIKeyHeader
from src.exception import CustomException


config = get_env()

logging.info("Getting the API_KEY")

API_KEY = config.get("API_KEY")
API_KEY_NAME = "X-API-Key"

logging.info("Extracting the entered API_KEY by the client from the header")

api_key_header = APIKeyHeader(name=API_KEY_NAME, auto_error=False)


def verify_api_key(api_key: str = Security(api_key_header)):
    
    """Verifying the api_key entered by the client matches the API_KEY given to it.

    Raises:
        HTTPException: If the api_key doesn't match with the API_KEY given to the client then unauthorized HTTP Exception (401) is generated.

    Returns:
        _type_: 16 bits hexadecimal code (API_KEY)
    """

    try:

        logging.info("Checking if the api_key entered by the client is None or != API_KEY")

        if api_key is None or api_key != API_KEY:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid or missing API Key"
            )
        return api_key
    
    except HTTPException:
        logging.exception("An HTTPException has occurred.")
        raise 

    except Exception as e:
        logging.exception("An error has occurred.")
        raise CustomException(e,sys)
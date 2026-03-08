import logging
import os
from datetime import datetime
    
LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)

LOG_FILE = os.path.join(LOG_DIR, f"log_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.log")

logging.basicConfig(filename=LOG_FILE, 
                    level=logging.INFO, 
                    format="[%(asctime)s] %(filename)s - %(levelname)s - %(name)s - %(message)s", 
                    datefmt="%Y-%m-%d %H:%M:%S"
    )


if __name__ == "__main__" :
    logging.info("Logging has started .") 
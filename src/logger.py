import logging
import os
from datetime import datetime 


LOG_FILE = f"{datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}.log"
#filename format fot the log file based on current timestamp
logs_path = os.path.join(os.getcwd(), "logs", LOG_FILE)
#creates a path to the log file by joining the current working directory 

os.makedirs(logs_path, exist_ok=True)
#creates the "logs" directory if it does not already exist
#logs dir contains is many subdirs in the datetime format

LOG_FILE_PATH = os.path.join(logs_path, LOG_FILE)

logging.basicConfig(
    filename=LOG_FILE_PATH,
    format="[ %(asctime)s ] %(lineno)d %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)
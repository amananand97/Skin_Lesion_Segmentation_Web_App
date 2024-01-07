import logging
import os
from datetime import datetime

# Creating a log file and using f-string naming file
LOG_FILE = f"{datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}.log"
# Creating Log folder where all the log files will append
logs_path=os.path.join(os.getcwd(),'logs', LOG_FILE)
os.makedirs(logs_path,exist_ok=True)

LOG_FILE_PATH = os.path.join(logs_path, LOG_FILE)

# Creating a basic configuration of log file/format
logging.basicConfig(
    filename=LOG_FILE_PATH,
    format= "[ %(asctime)s] %(lineno)d %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)

# if __name__=="__main__":
#     logging.info('Logging has started')
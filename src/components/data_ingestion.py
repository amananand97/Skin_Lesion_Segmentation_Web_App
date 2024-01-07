import os  # For operating system related functionalities
import sys  # For system-specific parameters and functions
import pandas as pd  # For data manipulation and analysis
from sklearn.model_selection import train_test_split  # For splitting data into training and testing sets
from dataclasses import dataclass  # For creating data classes

# Add the project root directory to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.append(project_root)

from src.exception import CustomException  # Custom exception handling
from src.logger import logging  # Custom logger
from src.components.data_transformation import DataTransformation  # Importing DataTransformation class
from src.components.data_transformation import DataTransformationConfig # Importing DataTransformationConfig class

from src.components.model_trainer import ModelTrainerConfig # Importing ModelTrainerConfig class
from src.components.model_trainer import ModelTrainer # Importing ModelTrainer class
@dataclass
class DataIngestionConfig:
    train_data_path: str = os.path.join('artifacts', 'train.csv')  # Default path for training data
    test_data_path: str = os.path.join('artifacts', 'test.csv')  # Default path for testing data
    raw_data_path: str = os.path.join('artifacts', 'data.csv')  # Default path for raw data

class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        logging.info("Entered the data ingestion method or component")  # Logging info message
        try:
            df = pd.read_csv('notebook/data/stud.csv')  # Reading data from CSV file
            logging.info('Imported the data as dataframe')  # Logging info message

            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path), exist_ok=True)  # Create directory if not exists

            df.to_csv(self.ingestion_config.raw_data_path, index=False, header=True)  # Save dataframe to CSV

            logging.info("Train test split initiated")  # Logging info message

            train_set, test_set = train_test_split(df, test_size=0.2, random_state=42)  # Split data into train and test sets

            train_set.to_csv(self.ingestion_config.train_data_path, index=False, header=True)  # Save train set to CSV
            test_set.to_csv(self.ingestion_config.test_data_path, index=False, header=True)  # Save test set to CSV

            logging.info("Ingestion of the data is completed")  # Logging info message

            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )

        except Exception as e:
            raise CustomException(e, sys)

if __name__ == "__main__":
    obj = DataIngestion()
    train_data, test_data = obj.initiate_data_ingestion()  # Initiate data ingestion

    data_transformation = DataTransformation()
    data_transformation.initiate_data_transformation(train_data, test_data)  # Initiate data transformation


if __name__ == "__main__":
    obj = DataIngestion()
    train_data, test_data = obj.initiate_data_ingestion()

    data_transformation = DataTransformation()
    train_arr,test_arr,_= data_transformation.initiate_data_transformation(train_data,test_data)

    modeltrainer = ModelTrainer()
    print(modeltrainer.initiate_model_trainer(train_arr,test_arr))
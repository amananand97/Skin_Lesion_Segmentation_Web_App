import os  # For operating system related functionalities
import sys  # For system-specific parameters and functions
from dataclasses import dataclass  # For creating data classes
import numpy as np  # For numerical computing
import pandas as pd  # For data manipulation and analysis
from sklearn.compose import ColumnTransformer  # For composing transformations
from sklearn.impute import SimpleImputer  # For imputing missing values
from sklearn.pipeline import Pipeline  # For creating a pipeline of transformations
from sklearn.preprocessing import OneHotEncoder, StandardScaler  # For preprocessing data

# Add the project root directory to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.append(project_root)

from src.exception import CustomException  # Custom exception handling
from src.logger import logging  # Custom logger
from src.utils import save_object  # Custom utility function

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path: str = os.path.join('artifacts', 'preprocessor.pkl')  # Default path for preprocessor object

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformer_object(self):
        try:
            numerical_columns = ["writing_score", "reading_score"]  # List of numerical columns
            categorical_columns = [  # List of categorical columns
                "gender",
                "race_ethnicity",
                "parental_level_of_education",
                "lunch",
                "test_preparation_course",
            ]

            num_pipeline = Pipeline(  # Creating a pipeline for numerical columns
                steps=[
                    ("imputer", SimpleImputer(strategy="median")),  # Imputing missing values using median
                    ("scaler", StandardScaler())  # Scaling numerical features
                ]
            )

            cat_pipeline = Pipeline(  # Creating a pipeline for categorical columns
                steps=[
                    ("imputer", SimpleImputer(strategy="most_frequent")),  # Imputing missing values using most frequent value
                    ("one_hot_encoder", OneHotEncoder()),  # Encoding categorical features using one-hot encoding
                    ("scaler", StandardScaler(with_mean=False))  # Scaling categorical features
                ]
            )

            logging.info(f"Categorical columns: {categorical_columns}")  # Logging info message
            logging.info(f"Numerical columns: {numerical_columns}")  # Logging info message

            preprocessor = ColumnTransformer(  # Creating a column transformer for preprocessing
                [
                    ("num_pipeline", num_pipeline, numerical_columns),  # Applying numerical pipeline to numerical columns
                    ("cat_pipelines", cat_pipeline, categorical_columns)  # Applying categorical pipeline to categorical columns
                ]
            )

            return preprocessor  # Returning the preprocessor object

        except Exception as e:
            raise CustomException(e, sys)

    def initiate_data_transformation(self, train_path, test_path):
        try:
            train_df = pd.read_csv(train_path)  # Reading train data from CSV
            test_df = pd.read_csv(test_path)  # Reading test data from CSV

            logging.info("Read train and test data completed")  # Logging info message
            logging.info("Obtaining preprocessing object")  # Logging info message

            preprocessing_obj = self.get_data_transformer_object()  # Getting the preprocessor object

            target_column_name = "math_score"  # Name of the target column
            numerical_columns = ["writing_score", "reading_score"]  # List of numerical columns

            input_feature_train_df = train_df.drop(columns=[target_column_name], axis=1)  # Extracting input features from train data
            target_feature_train_df = train_df[target_column_name]  # Extracting target feature from train data

            input_feature_test_df = test_df.drop(columns=[target_column_name], axis=1)  # Extracting input features from test data
            target_feature_test_df = test_df[target_column_name]  # Extracting target feature from test data

            logging.info("Applying preprocessing object on training dataframe and testing dataframe.")  # Logging info message

            input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df)  # Transforming input features of train data
            input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)  # Transforming input features of test data

            train_arr = np.c_[input_feature_train_arr, np.array(target_feature_train_df)]  # Combining input features and target feature of train data
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]  # Combining input features and target feature of test data

            logging.info("Saved preprocessing object.")  # Logging info message

            save_object(  # Saving the preprocessor object
                file_path=self.data_transformation_config.preprocessor_obj_file_path,  # File path to save the preprocessor object
                obj=preprocessing_obj  # Preprocessor object to be saved
            )

            return (  # Returning the transformed arrays and file path of preprocessor object
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path
            )
        except Exception as e:
            raise CustomException(e, sys)  # Raising custom exception with error message and system information

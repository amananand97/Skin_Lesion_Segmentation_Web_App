import os  # Importing the os module for operating system related functionalities
import sys  # Importing the sys module for system-specific parameters and functions
import dill  # Importing dill for object serialization
import pickle  # Importing pickle for object serialization
from sklearn.metrics import r2_score  # Importing r2_score for evaluating regression models
from sklearn.model_selection import GridSearchCV  # Importing GridSearchCV for hyperparameter tuning
import numpy as np  # Importing numpy for numerical computing
import pandas as pd  # Importing pandas for data manipulation

# Add the project root directory to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.append(project_root)  # Appending the project root directory to the Python path

from src.exception import CustomException  # Importing CustomException from src.exception
from src.logger import logging  # Importing the logging module from src.logger

def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)  # Getting the directory path of the file

        os.makedirs(dir_path, exist_ok=True)  # Creating the directory if it doesn't exist

        with open(file_path, 'wb') as file_obj:  # Opening the file in binary write mode
            dill.dump(obj, file_obj)  # Serializing and writing the object to the file

    except Exception as e:  # Handling any exceptions that occur
        raise CustomException(e, sys)  # Raising a CustomException with the error message and system information


def evaluate_models(X_train, y_train, X_test, y_test, models, param):
    try:
        report = {}  # Initializing an empty dictionary to store the evaluation report

        for i in range(len(list(models))):  # Iterating over the models
            model = list(models.values())[i]  # Getting the current model
            para = param[list(models.keys())[i]]  # Getting the parameters for the current model

            gs = GridSearchCV(model, para, cv=3)  # Creating a GridSearchCV object for hyperparameter tuning
            gs.fit(X_train, y_train)  # Fitting the GridSearchCV object to the training data

            model.set_params(**gs.best_params_)  # Setting the best parameters for the model
            model.fit(X_train, y_train)  # Fitting the model with the best parameters

            y_train_pred = model.predict(X_train)  # Making predictions on the training data
            y_test_pred = model.predict(X_test)  # Making predictions on the test data

            train_model_score = r2_score(y_train, y_train_pred)  # Calculating the R-squared score for the training data
            test_model_score = r2_score(y_test, y_test_pred)  # Calculating the R-squared score for the test data

            report[list(models.keys())[i]] = test_model_score  # Storing the test model score in the report dictionary

        return report  # Returning the evaluation report

    except Exception as e:  # Handling any exceptions that occur
        raise CustomException(e, sys)  # Raising a CustomException with the error message and system information


def load_object(file_path):
    try:
        with open(file_path, "rb") as file_obj:  # Opening the file in binary read mode
            return pickle.load(file_obj)  # Unpickling and returning the object from the file

    except Exception as e:  # Handling any exceptions that occur
        raise CustomException(e, sys)  # Raising a CustomException with the error message and system information

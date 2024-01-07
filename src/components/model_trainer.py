import os  # For operating system related functionalities
import sys  # For system-specific parameters and functions
from dataclasses import dataclass  # For creating data classes

# Importing machine learning models
from catboost import CatBoostRegressor
from sklearn.ensemble import (
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

# Add the project root directory to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.append(project_root)

# Custom imports
from src.exception import CustomException  # Custom exception handling
from src.logger import logging  # Custom logger
from src.utils import save_object, evaluate_models  # Custom utility functions

@dataclass
class ModelTrainerConfig:
    trained_model_file_path: str = os.path.join("artifacts", "model.pkl")  # Default path for trained model file

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()  # Initializing model trainer configuration

    def initiate_model_trainer(self, train_array, test_array):
        try:
            logging.info("Split training and test input data")  # Logging info message

            # Splitting input arrays into features and target
            X_train, y_train, X_test, y_test = (
                train_array[:, :-1],
                train_array[:, -1],
                test_array[:, :-1],
                test_array[:, -1],
            )

            # Dictionary of models to be evaluated
            models = {
                "Random Forest": RandomForestRegressor(),
                "Decision Tree": DecisionTreeRegressor(),
                "Gradient Boosting": GradientBoostingRegressor(),
                "Linear Regression": LinearRegression(),
                "XGBRegressor": XGBRegressor(),
                "CatBoosting Regressor": CatBoostRegressor(verbose=False),
                "AdaBoost Regressor": AdaBoostRegressor(),
            }

            # Parameters for hyperparameter tuning
            params = {
                "Decision Tree": {
                    'criterion': ['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                    'max_depth': [3, 5, 7],
                },
                "Random Forest": {
                    'n_estimators': [100, 200, 300],
                    'max_depth': [3, 5, 7],
                },
                "Gradient Boosting": {
                    'n_estimators': [100, 200, 300],
                    'learning_rate': [0.01, 0.1, 0.2],
                    'max_depth': [3, 5, 7],
                },
                "Linear Regression": {},
                "XGBRegressor": {
                    'n_estimators': [100, 200, 300],
                    'learning_rate': [0.01, 0.1, 0.2],
                    'max_depth': [3, 5, 7],
                },
                "CatBoosting Regressor": {
                    'iterations': [100, 200, 300],
                    'learning_rate': [0.01, 0.1, 0.2],
                    'depth': [3, 5, 7],
                },
                "AdaBoost Regressor": {
                    'n_estimators': [50, 100, 150],
                    'learning_rate': [0.01, 0.1, 0.2],
                }
            }

            # Evaluate models and get a report
            model_report = evaluate_models(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test, models=models, param=params)

            # Get the best model score and name
            best_model_score = max(sorted(model_report.values()))
            best_model_name = list(model_report.keys())[list(model_report.values()).index(best_model_score)]
            best_model = models[best_model_name]

            # If the best model score is less than 0.6, raise an exception
            if best_model_score < 0.6:
                raise CustomException("No best model is found")

            logging.info(f"Best found model on both training and testing dataset: {best_model_name}")  # Logging info message

            # Save the best model
            save_object(file_path=self.model_trainer_config.trained_model_file_path, obj=best_model)

            # Make predictions using the best model
            predicted = best_model.predict(X_test)

            # Calculate R-squared score
            r2_square = r2_score(y_test, predicted)

            return r2_square  # Return R-squared score

        except Exception as e:
            raise CustomException(e, sys)  # Raise custom exception with error message and system information

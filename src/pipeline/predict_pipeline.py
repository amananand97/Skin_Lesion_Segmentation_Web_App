import sys
import os
import pandas as pd

# Add the project root directory to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.append(project_root)

from src.exception import CustomException
from src.utils import load_object

class PredictPipeline:
    """Class for making predictions using a trained model."""

    def __init__(self):
        """Initialize an instance of PredictPipeline."""
        pass

    def predict(self, features):
        """Predict the target variable based on input features.

        Args:
            features (array-like): Input features for prediction.

        Returns:
            array: Predicted values.
        """
        try:
            # Define paths for model and preprocessor
            model_path = os.path.join("artifacts", "model.pkl")
            preprocessor_path = os.path.join("artifacts", "preprocessor.pkl")

            # Load model and preprocessor
            model = load_object(file_path=model_path)
            preprocessor = load_object(file_path=preprocessor_path)

            # Transform input features using the preprocessor
            data_scaled = preprocessor.transform(features)

            # Make predictions using the model
            preds = model.predict(data_scaled)
            return preds

        except Exception as e:
            # Raise custom exception if an error occurs
            raise CustomException(e, sys)


class CustomData:
    """Class to represent custom input data."""

    def __init__(self, gender, race_ethnicity, parental_level_of_education,
                 lunch, test_preparation_course, reading_score, writing_score):
        """Initialize the CustomData object with input values."""
        self.gender = gender
        self.race_ethnicity = race_ethnicity
        self.parental_level_of_education = parental_level_of_education
        self.lunch = lunch
        self.test_preparation_course = test_preparation_course
        self.reading_score = reading_score
        self.writing_score = writing_score

    def get_data_as_data_frame(self):
        """Convert input data to a pandas DataFrame.

        Returns:
            DataFrame: Custom input data as a DataFrame.
        """
        try:
            # Create a dictionary from input values
            custom_data_input_dict = {
                "gender": [self.gender],
                "race_ethnicity": [self.race_ethnicity],
                "parental_level_of_education": [self.parental_level_of_education],
                "lunch": [self.lunch],
                "test_preparation_course": [self.test_preparation_course],
                "reading_score": [self.reading_score],
                "writing_score": [self.writing_score],
            }

            # Convert the dictionary to a DataFrame
            return pd.DataFrame(custom_data_input_dict)

        except Exception as e:
            # Raise custom exception if an error occurs
            raise CustomException(e, sys)

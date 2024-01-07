# End-to-End Machine Learning Project with Prediction Pipeline

This project is an end-to-end machine learning project that includes data ingestion, data transformation, model training, and evaluation pipelines. It also includes a prediction pipeline using a Flask web app for making predictions with the trained model.

## Project Structure

The project structure is organized as follows:

- `application.py`: Flask application for serving predictions.
- `artifacts/`: Directory containing data and model artifacts.
- `catboost_info/`: Directory containing CatBoost model information.
- `logs/`: Directory containing log files.
- `notebook/`: Directory containing Jupyter notebooks for exploratory data analysis and model training.
- `requirements.txt`: File containing project dependencies.
- `setup.py`: Python package setup file.
- `src/`: Directory containing source code for the project.
  - `components/`: Directory containing reusable components for data ingestion, transformation, and model training.
  - `exception.py`: Module for custom exception handling.
  - `logger.py`: Module for logging.
  - `pipeline/`: Directory containing pipeline modules for prediction and training.
  - `utils.py`: Module containing utility functions.

## Usage

1. Install dependencies by running `pip install -r requirements.txt`.
2. **Data Ingestion:** To ingest data, run `python data_ingestion.py`. This script reads data from `notebook/data/stud.csv`, saves it to `artifacts/data.csv`, and splits it into training and testing sets saved as `artifacts/train.csv` and `artifacts/test.csv` respectively.
3. **Data Transformation:** To transform data, run `python data_transformation.py`. This script applies preprocessing transformations on the ingested data and saves the preprocessor object to `artifacts/preprocessor.pkl`.
4. **Model Training:** To train the model, run `python model_trainer.py`. This script trains various machine learning models, evaluates their performance, selects the best model with an R-squared score of at least 0.6, and saves it to `artifacts/model.pkl`.
5. **Prediction Pipeline:** The prediction pipeline is implemented in `predict_pipeline.py` in the `src/pipeline` directory. It contains classes for making predictions using a trained model and representing custom input data.
6. Run the Flask application using `python application.py`.
7. Access the web app at [http://localhost:5000](http://localhost:5000) to make predictions.


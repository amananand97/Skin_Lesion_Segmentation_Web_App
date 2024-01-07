from flask import Flask, request, render_template
import os
import sys
from src.pipeline.predict_pipeline import CustomData, PredictPipeline

# Add the project root directory to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.append(project_root)

application = Flask(__name__)

# Create a Flask application instance
app = application

# Route for the home page
@app.route('/')
def index():
    """Render the home page."""
    return render_template('index.html')

# Route for predicting data
@app.route('/predictdata', methods=['GET', 'POST'])
def predict_datapoint():
    """Predict the target variable based on input data."""
    if request.method == 'GET':
        return render_template('home.html')
    else:
        # Create CustomData object from form input
        data = CustomData(
            gender=request.form.get('gender'),
            race_ethnicity=request.form.get('ethnicity'),
            parental_level_of_education=request.form.get('parental_level_of_education'),
            lunch=request.form.get('lunch'),
            test_preparation_course=request.form.get('test_preparation_course'),
            reading_score=float(request.form.get('writing_score')),
            writing_score=float(request.form.get('reading_score'))
        )

        # Convert CustomData object to a DataFrame
        pred_df = data.get_data_as_data_frame()

        # Create PredictPipeline object and make predictions
        predict_pipeline = PredictPipeline()
        results = predict_pipeline.predict(pred_df)

        return render_template('home.html', results=results[0])

if __name__ == "__main__":
    # Run the Flask application
    app.run(host="0.0.0.0", port=5000)

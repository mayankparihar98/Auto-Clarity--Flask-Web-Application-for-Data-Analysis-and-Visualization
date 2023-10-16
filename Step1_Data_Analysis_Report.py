import os
import logging
from flask import Flask, render_template, redirect, url_for
from flask_wtf import FlaskForm
from wtforms import FileField, SubmitField
from werkzeug.utils import secure_filename
from wtforms.validators import InputRequired
import pandas as pd


# Set up logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

formatter = logging.Formatter('%(asctime)s : %(levelname)s : %(message)s')

file_handler = logging.FileHandler('app.log')
file_handler.setLevel(logging.INFO)
file_handler.setFormatter(formatter)

stream_handler = logging.StreamHandler()
stream_handler.setLevel(logging.DEBUG)
stream_handler.setFormatter(formatter)

logger.addHandler(file_handler)
logger.addHandler(stream_handler)

# Create a Flask app
app = Flask(__name__)
app.config['SECRET_KEY'] = 'supersecretkey'
app.config['UPLOAD_FOLDER'] = 'Dataset'

# Create a Flask form for uploading the file
class UploadFileForm(FlaskForm):
    """
    A Flask form for uploading a file.
    """
    file = FileField("File", validators=[InputRequired()]) # The file input field
    submit = SubmitField("Upload File") # The submit button

# Route for the home page and initial file upload form
@app.route('/', methods=['GET', 'POST'])
@app.route('/home', methods=['GET', 'POST'])
def home():
    form = UploadFileForm()
    
    if form.validate_on_submit():
        file = form.file.data
        filename = secure_filename(file.filename)
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        return redirect(url_for('analyze', filename=filename))  # Redirect to the 'analyze' route

    return render_template('upload_form.html', form=form)

# Route for analyzing the uploaded data and displaying results
@app.route('/analyze/<filename>', methods=['GET', 'POST'])
def analyze(filename):
    try:
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        data = pd.read_csv(file_path)
        logger.info("File read successfully")

        # Steps 1-5: Data Analysis and Visualization
        column_description = data.head().to_string()
        data_shape = data.shape
        data_info = data.info()
        data_describe = data.describe()
        missing_values = data.isnull().sum()

    except Exception as e:
        logger.error(f"Exception occurred: {str(e)}")
        return f"An exception occurred: {str(e)}"

    return render_template('results.html', data=data,
                           column_description=column_description, data_shape=data_shape,
                           data_info=data_info, data_describe=data_describe,
                           missing_values=missing_values, filename=filename)


if __name__ == '__main__':
    logging.basicConfig(filename='app.log', level=logging.INFO)
    app.run(debug=True)

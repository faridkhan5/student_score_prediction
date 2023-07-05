from flask import Flask, request, render_template
import numpy as np
import pandas as pd
import os

from src.pipeline.predict_pipeline import CustomData, PredictPipeline

application = Flask(__name__)

@application.route('/')
def index():
    return render_template('home.html')

@application.route('/predictdata', methods=['GET', 'POST'])
def predict_datapoint():
    if request.method == 'GET':
        return render_template('home.html')
    else:
        gender = request.form.get('gender')
        race_ethnicity=request.form.get('race_ethnicity')
        parental_level_of_education=request.form.get('parental_level_of_education')
        lunch=request.form.get('lunch')
        test_preparation_course=request.form.get("test_preparation_course")
        reading_score=float(request.form.get("reading_score"))
        writing_score=float(request.form.get("writing_score"))
        
        data = CustomData(gender=gender, race_ethnicity=race_ethnicity, parental_level_of_education=parental_level_of_education, lunch=lunch, test_preparation_course=test_preparation_course, reading_score=reading_score, writing_score=writing_score)

        pred_df = data.get_data_as_data_frame()
        print(pred_df)

        predict_pipeline = PredictPipeline()
        results = predict_pipeline.predict(pred_df)
        return render_template('result.html', results=results[0], gender=gender, race_ethnicity=race_ethnicity, parental_level_of_education=parental_level_of_education, lunch=lunch, test_preparation_course=test_preparation_course, reading_score=reading_score, writing_score=writing_score)

if __name__=="__main__":
    port = int(os.environ.get("PORT", 80))
    application.run(host="0.0.0.0")
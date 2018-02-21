from flask import Flask, jsonify, request
import pandas as pd
from pipeline import *
from settings import *
from sklearn.externals import joblib

app = Flask(__name__)

@app.route('/')

def home():
    return "Titanic predictive API"

@app.route('/predict', methods = ['POST'])

def predict():
    try:
        json_ = request.get_json()
        query_df = pd.DataFrame(json_)
        X = pipeline.fit_transform(query_df)
        prediction = model.predict(X)
    except Exception as e:
        raise e
    return jsonify({'prediction': prediction.tolist()})

if __name__ == '__main__':
    model_path = MODELS_DIR + "/" + model_file
    pipeline_path = MODELS_DIR + "/" + pipeline_file
    
    model = joblib.load(model_path)
    pipeline = joblib.load(pipeline_path)
    app.run(port=8080)
   
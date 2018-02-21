Titanic, Kaggle Challenge, Prediction
-----------------------

A simple predictive API that use a logistic regression model to predict which Tinanic passenger had survived the tragedy or not. Train and test data can be downloaded on Kaggle Titanic challenge https://www.kaggle.com/c/titanic

In this project a data pipeline, from sklearn package, is used to clean data before it is feeded to train a logistic model. The same pipeline is also used to clean test data.

Installation
----------------------

### Download the data

* Clone this repo to your computer.
* Get into the folder using `cd titanic-api`.
* Run `mkdir data`.
* Switch into the `data` directory using `cd data`.
* Run `mkdir raw`.
* Switch into the `data` directory using `cd raw`.
* Download train.csv and test.csv on Kaggle Titanic challenge into raw data

### Train logistic model
* Run `python train_model.py` to clean the `train` dataset and train a logistic model using that dataset. The pipeline and model is saved in `models` directory

### Run API 
* Run `python app.py` to run predictive API. The API will be run on http://127.0.0.1:8080/ 
* A post request can be sent to http://127.0.0.1:8080/predict to get predicted value 

### Run a simple test
* Run `python test.py` to send a post request to our API. The result should be printed on the terminal window

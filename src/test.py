import requests
import json
import pandas as pd 

X_test = pd.read_csv("data/test.csv")

r = requests.post('http://127.0.0.1:8080/predict'
                  #, data = json.loads(X_test.loc[:1,].to_json())
                  , json = json.loads(X_test.to_json())
                  )
print(r.json())

from settings import *
import requests
import json
import pandas as pd 

test_data_path = RAW_DATA_DIR + "/" + test_file
X_test = pd.read_csv(test_data_path)

r = requests.post('http://127.0.0.1:8080/predict'
                  , json = json.loads(X_test.to_json())
                  )
print(r.json())

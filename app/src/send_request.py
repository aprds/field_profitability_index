import json
import requests
import pandas as pd

#data = pd.read_csv('../data/prediction_1.csv').to_dict()

data = {
            "fluid": "Oil",
            "field_name": "AP",
            "operator": "PERTAMINA",
            "project_status": "OFFSHORE",
            "inplace": 1.5,
            "depth": 7900,
            "temp": 125.4,
            "poro": 0.237,
            "perm": 111.265,
            "saturate": 0.582,
            "api_dens": 38,
            "visc": 8.38,
            "avg_fluid_rate": 214,
            "location": "Jawa Barat",
            "region": "Jawa"
  }

data_json = json.dump(data)
url_string = f'http://localhost:8000/predict'

r = requests.get(url_string, data=data_json)

print(r.json())
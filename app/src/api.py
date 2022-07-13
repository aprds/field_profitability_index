import pandas as pd
import joblib
import yaml
from preprocessing import preprocess
from feature_engineering import feature_eng
from utils import read_yaml
from prediction import main_predict
from fastapi import FastAPI, Form, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from tqdm import tqdm

tqdm.pandas()

PREPROCESSING_CONFIG_PATH = "../config/preprocessing_config.yaml"
FEATURE_ENGINEERING_CONFIG_PATH = "../config/feature_engineering_config.yaml"

model = joblib.load('../model/fitted_model.pkl')

param_feat_eng = read_yaml(FEATURE_ENGINEERING_CONFIG_PATH)
param_preprocess = read_yaml(PREPROCESSING_CONFIG_PATH)

app = FastAPI()

origins = ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def res_constructor(predict, proba):
    res = {'result': predict, 'proba': proba}
    return res

@app.post("/predict/")

def predict_api(fluid: str =Form(), field_name: str =Form(), operator: str =Form(), project_status: str =Form(), 
                inplace: float =Form(), depth: float =Form(), temp: float =Form(), poro: float =Form(), 
                perm:float =Form(), saturate: float =Form(), api_dens: float =Form(), visc: float =Form(), 
                avg_fluid_rate: float =Form(), location: str =Form(), region: str =Form()):
    
    try:

        forming = {'fluid': fluid, 'field_name': field_name, 'operator': operator, 'project_status': project_status,
                    'inplace': inplace, 'depth': depth, 'temp': temp, 'poro': poro,
                    'perm': perm, 'saturate': saturate, 'api_dens': api_dens, 'visc': visc,
                    'avg_fluid_rate': avg_fluid_rate, 'location': location, 'region': region}

        df = pd.DataFrame([forming])

        predict, proba = main_predict(df, model, param_preprocess, param_feat_eng)
        res = res_constructor(predict, proba)
        return res

    except Exception as e:
        return {'result': "", 'proba': "", 'message': str(e)}
    
    
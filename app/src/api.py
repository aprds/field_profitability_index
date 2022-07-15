from email.policy import default
from unicodedata import decimal
import pandas as pd
import numpy as np
import joblib
import yaml
import json
import uvicorn
from preprocessing import preprocess
from feature_engineering import feature_eng
from utils import read_yaml
from prediction import main_predict
from fastapi import FastAPI, Form, Header, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.encoders import jsonable_encoder
from fastapi.responses import ORJSONResponse, JSONResponse
import os
from tqdm import tqdm
from pydantic import BaseModel

tqdm.pandas()

app = FastAPI(title= 'Field Profitability Index Prediction', description= 'Predict the field profitability using its nearby field information',
                version='1.0', default_response_class=ORJSONResponse)

origins = ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/predict/")
async def get_prediction(
    fluid: str =Query(..., description='Enter fluid type: Oil | Gas | Oil-Gas',),
    field_name: str =Query(..., description='Enter field name',),
    operator: str =Query(..., description='Enter operator name:',),
    project_status: str =Query(..., description='Enter project status: ONSHORE | OFFSHORE | BOTH',),
    inplace: float =Query(..., description='Enter inplace (MMBO.E):',),
    depth: float =Query(..., description='Enter depth (feet):',),
    temp: float =Query(..., description='Enter temperature (F):',),
    poro: float =Query(..., description='Enter porosity (fraction):',),
    perm: float =Query(..., description='Enter permeability (md):',),
    saturate: float =Query(..., description='Enter saturation (fraction):',),
    api_dens: float =Query(..., description='Enter density: API | Sg',), 
    visc: float =Query(..., description='Enter viscosity (cp):',),
    avg_fluid_rate: float =Query(..., description='Enter average fluid rate (BOPD.E):',),
    location: str =Query(..., description='Enter field location: Aceh | Jambi | Jawa Barat | Jawa Tengah | Jawa Timur | Kalimantan Selatan | Kalimantan Tengah | Kalimantan Timur | Kalimantan Utara | Laut Cina Utara | Laut Jawa | Laut Natuna | Laut Natuna Utara | Laut Seram | Laut Timor | Maluku | Papua Barat | Riau | Selat Makasar | Selat Malaka | Sulawesi Barat | Sulawesi Selatan | Sulawesi Tengah | Sulawesi Tengah (offshore) | Sumatera Barat | Sumatera Selatan | Sumatera Utara | Teluk Berau',),
    region:str =Query(..., description='Enter field region: Jawa | Kalimantan | Sumatera | Timur',),):
    
    try:
        PREPROCESSING_CONFIG_PATH = "../config/preprocessing_config.yaml"
        FEATURE_ENGINEERING_CONFIG_PATH = "../config/feature_engineering_config.yaml"

        model = joblib.load('../model/fitted_model.pkl')

        param_feat_eng = read_yaml(FEATURE_ENGINEERING_CONFIG_PATH)
        param_preprocess = read_yaml(PREPROCESSING_CONFIG_PATH)

        def res_constructor(predict, proba):
            if proba <= 0.5: # Return proba to represent models confident level
                proba = 1 - proba

            else:
                proba
            
            res = {'result': predict, 'proba': proba}

            return res

        
        class NpEncoder(json.JSONEncoder):
            def default(self, obj):
                if isinstance(obj, np.integer):
                    return int(obj)
                if isinstance(obj, np.floating):
                    return float(obj)
                if isinstance(obj, np.ndarray):
                    return obj.tolist()

                return json.JSONEncoder.default(self, obj)


        col = ['fluid','field_name','operator','project_status',
                'inplace','depth','temp','poro',
                'perm','saturate','api_dens','visc',
                'avg_fluid_rate','location','region']

        val = [[fluid,field_name,operator,project_status,
                inplace,depth,temp,poro,
                perm,saturate,api_dens,visc,
                avg_fluid_rate,location,region]]

        df = pd.DataFrame(val, columns=col)

        predict, proba = main_predict(df, model, param_preprocess, param_feat_eng)
        res = res_constructor(predict, proba)

        return json.dumps(res, cls=NpEncoder)

    except Exception as e:
        return {'result': "", 'proba': "", 'message': str(e)}
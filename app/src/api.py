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
from pydantic import BaseModel

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

class Fld_indx(BaseModel):
    fluid: str
    field_name: str 
    operator: str
    project_status: str
    inplace: float
    depth: float
    temp: float
    poro: float
    perm: float
    saturate: float
    api_dens: float
    visc: float
    avg_fluid_rate: float
    location: str
    region: str


@app.get("/predict/")
async def get_prediction(feat: Fld_indx):
    fluid= feat.fluid
    field_name= feat.field_name
    operator= feat.operator
    project_status= feat.project_status
    inplace= feat.inplace
    depth= feat.depth
    temp= feat.temp
    poro= feat.poro
    perm= feat.perm
    saturate= feat.saturate
    api_dens= feat.api_dens
    visc= feat.visc
    avg_fluid_rate= feat.avg_fluid_rate
    location= feat.location
    region= feat.region

    
    try:
        col = ['fluid','field_name','operator','project_status',
                'inplace','depth','temp','poro',
                'perm','saturate','api_dens','visc',
                'avg_fluid_rate','location','region']

        val = [fluid,field_name,operator,project_status,
                inplace,depth,temp,poro,
                perm,saturate,api_dens,visc,
                avg_fluid_rate,location,region]

        df = pd.DataFrame(val, columns=col)

        predict, proba = main_predict(df, model, param_preprocess, param_feat_eng)
        res = res_constructor(predict, proba)

        return res

    except Exception as e:
        return {'result': "", 'proba': "", 'message': str(e)}
    
    
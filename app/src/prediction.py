import joblib
import pandas as pd
from preprocessing import preprocess

from utils import read_yaml

PREPROCESSING_CONFIG_PATH = "../config/preprocessing_config.yaml"
FEATURE_ENGINEERING_CONFIG_PATH = "../config/feature_engineering_config.yaml"

tfidf_vect = joblib.load("../output/vectorizer.pkl")
model = joblib.load('../model/mantab_model.pkl')
threshold = joblib.load('../model/threshold.pkl')


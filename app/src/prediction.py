from xml.sax.handler import feature_string_interning
import joblib
import pandas as pd
from preprocessing import preprocess
from feature_engineering import feature_eng

from utils import read_yaml

PREPROCESSING_CONFIG_PATH = "../config/preprocessing_config.yaml"
FEATURE_ENGINEERING_CONFIG_PATH = "../config/feature_engineering_config.yaml"

model = joblib.load('../model/fitted_model.pkl')

def main_predict(df, model, param_preprocess, param_feat_eng):

    df_preprocessed = preprocess(df, param_preprocess)
    df_feature_eng = feature_eng(df_preprocessed, param_feat_eng)

    proba = model.predict_proba(df_feature_eng)[:, 1]
    predict = 1 if proba > 0.5 else 0

    return predict, proba


param_feat_eng = read_yaml(FEATURE_ENGINEERING_CONFIG_PATH)
param_preprocess = read_yaml(PREPROCESSING_CONFIG_PATH)


if __name__ == "__main__":
    while(1):
        print("Membaca Data Set Prediksi:")
        df = pd.read_csv('data/prediction.csv')
        predict, proba = main_predict(df, model, param_preprocess, param_feat_eng)
        print("Hasil prediksi\t:", predict)
        print("Probabilitas\t:", proba[0], "\n\n")
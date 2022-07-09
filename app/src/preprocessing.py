from tqdm import tqdm
import joblib
import numpy as np
import pandas as pd

tqdm.pandas()

from utils import read_yaml

PREPROCESSING_CONFIG_PATH = "../config/preprocessing_config.yaml"


def load_split_data(params):
    """
    Loader for splitted data.
    
    Args:
    - params(dict): preprocessing params.
    
    Returns:
    - x_train(DataFrame): inputs of train set.
    - x_valid(DataFrame): inputs of valid set.
    - x_test(DataFrame): inputs of test set.
    """

    x_train = joblib.load(params["out_path"]+"x_train.pkl")
    x_valid = joblib.load(params["out_path"]+"x_valid.pkl")
    x_test = joblib.load(params["out_path"]+"x_test.pkl")

    return x_train, x_valid, x_test


def depth(df_in, do=True):
    """
    Function for depth 0 & NaN imputation
    """
    df = df_in.copy()  # Avoid modifying the main dataframe
    if do:
        jawa = df.loc[(df.depth != 0) & (df.region == 'Jawa')].groupby('project_status')['depth'].median()
        kalimantan = df.loc[(df.depth != 0) & (df.region == 'Kalimantan')].groupby('project_status')['depth'].median()
        sumatera = df.loc[(df.depth != 0) & (df.region == 'Sumatera')].groupby('project_status')['depth'].median()
        timur = df.loc[(df.depth != 0) & (df.region == 'Timur')].groupby('project_status')['depth'].median()
        
        df.loc[((df.depth == 0) | (df.depth.isna())), 'depth'] = df.apply(lambda row: jawa[row['project_status']] if (row['region'] == 'Jawa')  else
                    kalimantan[row['project_status']] if (row['region'] == 'Kalimantan') else
                    sumatera[row['project_status']] if (row['region'] == 'Sumatera') else
                    timur[row['project_status']] if (row['region'] == 'Timur') else row['depth'], axis=1)
       
    return df


def temperature(df_in, do=True):
    """
    Main function for temperature 0 & NaN imputation
    """
    df = df_in.copy()  # Avoid modifying the main dataframe
    if do:
        df['temp'] = df.apply(lambda row: np.exp(1.3489 + np.log(row['depth'])*0.3742) if ((np.isnan(row['temp'])) 
                    | (row['temp'] == 0)) else row['temp'], axis=1)

    return df


def field_name(df_in, do=True):
    """
    Main function for missing field name imputation (using operator first name)
    """
    df = df_in.copy()  # Avoid modifying the main dataframe
    if do:
        df['field_name'] = df.apply(lambda row: 'x_field' if (pd.isna(row['field_name']))
                    else row['field_name'], axis=1)

    return df


def preprocess(df_in, params):
    """
    A function to execute the preprocessing steps.
    
    Args:
    - df_in(DataFrame): Input dataframe
    - params(dict): preprocessing parameters
    
    Return:
    - df(DataFrame): preprocessed data
    """
    df = df_in.copy()
    df = depth(df, params['depth_in'])
    df = temperature(df, params['temp_in'])
    df = field_name(df, params['field_name_in'])

    return df


def main_prep(x_train,x_valid,x_test, params):
    x_list = [x_train,x_valid,x_test]

    x_preprocessed = []
    for x in tqdm(x_list):
        temp = preprocess(x, params)
        x_preprocessed.append(temp)

    name = ['train','valid','test']
    for i,x in tqdm(enumerate(x_preprocessed)):
        joblib.dump(x, f"{params['out_path']}x_{name[i]}_preprocessed.pkl")
    
    return x_preprocessed
    
    
if __name__ == "__main__":
    params_preprocess = read_yaml(PREPROCESSING_CONFIG_PATH)
    x_train, x_valid, x_test = load_split_data(params_preprocess)
    x_preprocessed_list = main_prep(x_train, x_valid, x_test, params_preprocess)
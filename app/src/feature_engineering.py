from tqdm import tqdm
import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder

tqdm.pandas()

from utils import read_yaml

FEATURE_ENGINEERING_CONFIG_PATH = "../config/feature_engineering_config.yaml"

def load_preprocessed_data(params):
    """
    Loader for preprocessed data.
    
    Args:
    - params(dict): preprocessing params.
    
    Returns:
    - list_of_preprocessed(List): list of preprocessed data.
    """
    name = ['train','valid','test']
    list_of_preprocessed = []
    for i in name:
        path = f"{params['out_path']}x_{i}_preprocessed.pkl"
        temp = joblib.load(path)
        list_of_preprocessed.append(temp)

    return list_of_preprocessed


def prospect_ranking(df_in, do=True):
    """
    function to binning the PI into rank (0 & 1)

    Args:
    - df_in(DataFrame): Input data
    """
    df = df_in.copy()
    if do:
        df['prospect_rank'] = df.apply(lambda row: 1 if ((row['project_level'] == 'E0. On Production') | (row['project_level'] == 'E1. Production on Hold')
                        | (row['project_level'] == 'E2. Under Development') | (row['project_level'] == 'E8. Further Development Not Viable')) else 0
                        if ((row['project_level'] == 'E7. Production Not Viable') | (row['project_level'] == 'X3. Development Not Viable')) else 1, axis=1)

    return df


def operatorship(df_in, do=True):
    """
    function to change operator into Pertamina & Non-Pertamina

    Args:
    - df_in(DataFrame): Input data
    """
    df = df_in.copy()
    if do:
        df['operator'] = df.apply(lambda row: 'PERTAMINA' if ('PERTAMINA' in row['operator']) else 'NON_PERTAMINA', axis=1)   

    return df


def unit_conv(df_in, do=True):
    """
    function for unit convertion

    Args:
    - df_in('inplace'): on MMBO.E
    - df_in('avg_fluid_rate): on BOPD.E
    """
    df = df_in.copy()
    if do:
        df['inplace'] = df.apply(lambda row: row['inplace']/1000 if (row['fluid'] == 'Oil') | (row['fluid'] == 'Oil_Gas') 
                        else row['inplace']/5.6, axis=1) #Inplace conversion

        df['avg_fluid_rate'] = df.apply(lambda row: ((row['avg_fluid_rate']/1000)/5.6)*1000000 if row['fluid'] == 'Gas'
                        else row['avg_fluid_rate'], axis=1) #Rate conversion

    return df


def transformation(df_in, do=True):
    """
    function for data transformatio, focus on ordinal data to become one-hot-encoded

    Args:
    - df_in(DataFrame): Input data
    """
    df = df_in.copy()
    if do:
        num_df = df._get_numeric_data()
        cat_df = num_df.drop(columns=num_df.columns)

        cat_encoder = OneHotEncoder()

        hot_cat_df = cat_encoder.fit_transform(cat_df)

        cat_columns = ['Gas', 'Oil', 'Oil-Gas',
            'NON_PERTAMINA', 'PERTAMINA', 
            'BOTH', 'OFFSHORE', 'ONSHORE', 
            'Aceh', 'Jambi', 'Jawa Barat', 'Jawa Tengah', 'Jawa Timur',
            'Kalimantan Selatan', 'Kalimantan Tengah', 'Kalimantan Timur',
            'Kalimantan Utara', 'Laut Cina Utara', 'Laut Jawa', 'Laut Natuna',
            'Laut Natuna Utara', 'Laut Seram', 'Laut Timor', 'Maluku',
            'Papua Barat', 'Riau', 'Selat Makasar', 'Selat Malaka',
            'Sulawesi Barat', 'Sulawesi Selatan', 'Sulawesi Tengah',
            'Sulawesi Tengah (offshore)', 'Sumatera Barat', 'Sumatera Selatan',
            'Sumatera Utara', 'Teluk Berau', 
            'Jawa', 'Kalimantan', 'Sumatera', 'Timur']

        tr_df_cat = pd.DataFrame(hot_cat_df, columns=cat_columns)
        tr_feat_df = pd.concat((num_df, tr_df_cat), axis=1)

    return tr_feat_df


def main_feat(x_preprocessed_list, params):
    """
    Main function for feature engineering
    """
    x_train_preprocessed, x_valid_preprocessed, x_test_preprocessed = x_preprocessed_list

    df_train_feat = prospect_ranking(x_train_preprocessed, params['pros_rank'])
    df_valid_feat = prospect_ranking(x_valid_preprocessed, params['pros_rank'])
    df_test_feat = prospect_ranking(x_test_preprocessed, params['pros_rank'])

    df_train_feat = operatorship(df_train_feat, params['operator'])
    df_valid_feat = operatorship(df_valid_feat, params['operator'])
    df_test_feat = operatorship(df_test_feat, params['operator'])

    df_train_feat = df_train_feat.drop(columns=['field_name', 'project_level', 'cap_cost', 'opr_cost','total_cost', 'NPV', 'PI'])
    df_valid_feat = df_valid_feat.drop(columns=['field_name', 'project_level', 'cap_cost', 'opr_cost','total_cost', 'NPV', 'PI'])
    df_test_feat = df_test_feat.drop(columns=['field_name', 'project_level', 'cap_cost', 'opr_cost','total_cost', 'NPV', 'PI'])

    df_train_feat = unit_conv(df_train_feat, params['conv'])
    df_valid_feat = unit_conv(df_valid_feat, params['conv'])
    df_test_feat = unit_conv(df_test_feat, params['conv'])

    df_train_feat = transformation(df_train_feat, params['transformed'])
    df_valid_feat = transformation(df_valid_feat, params['transformed'])
    df_test_feat = transformation(df_test_feat, params['transformed'])


    joblib.dump(df_train_feat, f"{params['out_path']}x_train_feat.pkl")
    joblib.dump(df_valid_feat, f"{params['out_path']}x_valid_feat.pkl")
    joblib.dump(df_test_feat, f"{params['out_path']}x_test_feat.pkl")
    
    return df_train_feat, df_valid_feat, df_test_feat


if __name__ == "__main__":
    param_feat = read_yaml(FEATURE_ENGINEERING_CONFIG_PATH)
    x_preprocessed_list = load_preprocessed_data(param_feat)
    x_train_vect, x_valid_vect, x_test_vect = main_feat(x_preprocessed_list, param_feat)
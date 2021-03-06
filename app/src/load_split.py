import pandas as pd
from tqdm import tqdm
import joblib
from sklearn.model_selection import train_test_split

tqdm.pandas()

from utils import read_yaml

LOAD_SPLIT_CONFIG_PATH = "../config/load_split_config.yaml"


def target_added(df_in, target):

    df = df_in.copy()
    df[target] = df.apply(lambda row: 1 if ((row['project_level'] == 'E0. On Production') | (row['project_level'] == 'E1. Production on Hold')
                        | (row['project_level'] == 'E2. Under Development') | (row['project_level'] == 'E8. Further Development Not Viable')) else 0
                        if ((row['project_level'] == 'E7. Production Not Viable') | (row['project_level'] == 'X3. Development Not Viable')) else 1, axis=1)

    return df
    

def split_xy(df_in, x_col, y_col):
    """
    Splitting x and y variables.
    
    Args:
    - df(DataFrame): initial input dataframe
    - x_col(list): List of x variable columns
    - y_col(list): List of y variable columns
    
    Returns:
    - feat (DataFrame): Dataframe contains x columns/ fetures
    - target (DataFrame): Dataframe contains y columns/ target
    """
    df = df_in.copy()
    feat = df.loc[:, x_col]
    target = df.loc[:, y_col]

    return feat, target


def get_stratify_col(y):
    """
    Splitting x and y variables.
    
    Args:
    - y(DataFrame): DataFrame contains target variables and id
    - stratify_col(str): column name of the reference column.
    
    Returns:
    - stratification: Dataframe contains column that will be used as stratification reference
    """
    stratification = y
    
    return stratification


def run_split_data(x, y, stratify_col=None, TEST_SIZE=0.2):
    """
    Splitting x and y variables.
    
    Args:
    - y(DataFrame): DataFrame contains predictor variables and id
    - y(DataFrame): DataFrame contains target variables and id
    - stratify_col(str): column name of the reference column.
    - TEST_SIZE(float): Size of the test and validation dataset size.
    
    Returns:
    - x_blabla(DataFrame): X variables for train/valid/test dataset
    - y_blabla(DataFrame): Y variables for train/valid/test dataset
    """
    strat_train = get_stratify_col(y)
    x_train, x_test, y_train, y_test = train_test_split(x, y,
                                       stratify = strat_train,
                                       test_size= TEST_SIZE*2,
                                       random_state= 42)
    
    strat_test = get_stratify_col(y_test)
    x_valid, x_test, y_valid, y_test = train_test_split(x_test, y_test,
                                       stratify = strat_test,
                                       test_size= 0.5,
                                       random_state= 42)
    
    return x_train, y_train, x_valid, y_valid, x_test, y_test


def main_load(params):
    df = pd.read_excel(params['file_loc'])
    df = target_added(df, params['y_col'])
    x_all, y_all = split_xy(df, params['x_col'], params['y_col'])
    x_train, y_train,x_valid, y_valid,x_test, y_test = run_split_data(x_all, y_all, 
                                                                      y_all, 
                                                                      params['test_size'])
    joblib.dump(x_train, params["out_path"]+"x_train.pkl")
    joblib.dump(y_train, params["out_path"]+"y_train.pkl")
    joblib.dump(x_valid, params["out_path"]+"x_valid.pkl")
    joblib.dump(y_valid, params["out_path"]+"y_valid.pkl")
    joblib.dump(x_test, params["out_path"]+"x_test.pkl")
    joblib.dump(y_test, params["out_path"]+"y_test.pkl")
    
    return x_train, y_train, x_valid, y_valid, x_test, y_test

if __name__ == "__main__":
    params = read_yaml(LOAD_SPLIT_CONFIG_PATH)
    x_train, y_train, x_valid, y_valid, x_test, y_test = main_load(params)
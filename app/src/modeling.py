from ast import Raise
from unittest import result
import numpy as np
import pandas as pd
from sklearn.decomposition import SparsePCA
from sympy import hyper
from tqdm import tqdm
import joblib
import time
from sklearn.calibration import CalibratedClassifierCV
from sklearn.utils.class_weight import compute_class_weight
import xgboost as xgb
from sklearn.model_selection import RepeatedStratifiedKFold
from scipy.stats import loguniform
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import classification_report, roc_auc_score, f1_score, make_scorer, precision_score, accuracy_score, recall_score, balanced_accuracy_score
from hyperopt import STATUS_OK, Trials, fmin, hp, tpe
from sklearn.model_selection import KFold, cross_val_score
from hyperopt.pyll.base import scope
from hyperopt.pyll.stochastic import sample


tqdm.pandas()

from utils import read_yaml

MODELING_CONFIG_PATH = "../config/modeling_config.yaml"

def load_fed_data():
    """
    Loader for feature engineered data.
    Args:
    - params(dict): modeling params.
    Returns:
    - x_train(DataFrame): inputs of train set.
    - y_train(DataFrame): target of train set.
    - x_valid(DataFrame): inputs of valid set.
    - y_valid(DataFrame): terget of valid set.
    """

    x_train_path = "../output/x_train_feat.pkl"
    y_train_path = "../output/y_train.pkl"
    x_valid_path = "../output/x_valid_feat.pkl"
    y_valid_path = "../output/y_valid.pkl"
    x_train = joblib.load(x_train_path)
    y_train = joblib.load(y_train_path)
    x_valid = joblib.load(x_valid_path)
    y_valid = joblib.load(y_valid_path)

    return x_train, y_train, x_valid, y_valid


def hyperoptimize(x_train, y_train, x_valid, y_valid):

    trials = Trials()

    def objective_CV(space):

        clf = xgb.XGBClassifier(n_estimators = int(space['n_estimators']), max_depth = int(space['max_depth']), gamma = space['gamma'],
                        reg_alpha = int(space['reg_alpha']), reg_lambda = space['reg_lambda'], subsample = space['subsample'],
                        eta = space['eta'], min_child_weight = int(space['min_child_weight']), seed = 0, 
                        eval_metric = 'auc', objective = 'binary:logistic'
                        )

        clf_train = clf.fit(x_train, y_train)
        clf_valid = clf.fit(x_valid, y_valid)

        y_valid_pred = clf_train.predict_proba(x_valid)[:,1]
        y_train_pred = clf_valid.predict_proba(x_train)[:,1]

        score = np.mean([roc_auc_score(y_valid, y_valid_pred), roc_auc_score(y_train, y_train_pred)])

        print(f'roc_auc_score:', score)

        return {'loss': -score, 'status': STATUS_OK}

    best_hyperparams = fmin(objective_CV,
                        space = {'n_estimators' : hp.quniform('n_estimators', 5, 1000, 100),
                                'max_depth' : hp.quniform('max_depth', 2, 20, 1),
                                'gamma' : hp.uniform('gamma', 0, 2),
                                'reg_alpha' : hp.quniform('reg_alpha', 0, 50, 1),
                                'reg_lambda' : hp.uniform('reg_lambda', 0, 50),
                                'subsample' : hp.uniform('subsample', 0.5, 1),
                                'eta' : hp.uniform('eta', 0.01, 0.5),
                                'min_child_weight' : hp.quniform('min_child_weight', 0, 10, 1)},
                        algo = tpe.suggest,
                        max_evals = 100,
                        trials = trials)

    return best_hyperparams


def model_fit(best_hyperparams, x_train, y_train):

    clf_XGB = xgb.XGBClassifier(random_state=42, 
                            booster='gbtree', 
                            eta=best_hyperparams['eta'], 
                            gamma=best_hyperparams['gamma'], 
                            subsample=best_hyperparams['subsample'], 
                            max_depth=int(best_hyperparams['max_depth']), 
                            reg_lambda=best_hyperparams['reg_lambda'], 
                            reg_alpha=int(best_hyperparams['reg_alpha']),
                            grow_policy='depthwise',
                            n_estimators=int(best_hyperparams['n_estimators']),
                            min_child_weight =int(best_hyperparams['min_child_weight']),
                            seed=0,
                            eval_metric='auc',
                            objective='binary:logistic')

    fitted_model = clf_XGB.fit(x_train, y_train)

    return fitted_model


def classif_report(fitted_model, x_test, y_test):

    y_pred = fitted_model.predict(x_test)

    print('Test Data: accuracy_score: %s' % accuracy_score(y_test, y_pred))
    print('Test Data: balanced_accuracy_score: %s' % balanced_accuracy_score(y_test, y_pred))
    print('Test Data: precision_score: %s' % precision_score(y_test, y_pred))
    print('Test Data: recall_score: %s' % recall_score(y_test, y_pred))
    print('Test Data: roc_auc_score:', roc_auc_score(y_test, fitted_model.predict_proba(x_test)[:,1]))

    return y_pred


def validate(fitted_model, x_valid, y_valid):
    """
    Validate model
    Args:
        - x_valid(DataFrame): Validation independent variables
        - y_valid(DataFrame): Validation Dependent variables
        - model_fitted(callable): Sklearn / imblearn fitted model
        
    Return:
        - report_model: sklearn model report
        - model_calibrated(callable): Calibrated model
        - best_threshold(float): Best threshold
    """

    return classif_report(fitted_model, x_valid, y_valid)


def main(x_train, y_train, x_valid, y_valid, params):
    
    # Train
    t0 = time.time()
    best_hyperparams = hyperoptimize(x_train, y_train, x_valid, y_valid)
    fitted_model = model_fit(best_hyperparams, x_train, y_train)
    elapsed_time = time.time() - t0

    print(f'elapsed time: {elapsed_time} s \n')


    # Validate
    validate(fitted_model, x_valid, y_valid)


    joblib.dump(fitted_model, params['out_path']+'fitted_model.pkl')
    
    return fitted_model


if __name__ == "__main__":
    param_model = read_yaml(MODELING_CONFIG_PATH)
    x_train, y_train, x_valid, y_valid = load_fed_data()
    fitted_model = main(x_train, y_train, x_valid, y_valid, param_model)
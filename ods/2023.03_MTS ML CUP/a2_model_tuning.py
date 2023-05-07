import numpy as np
import pandas as pd

import optuna
from optuna.samplers import TPESampler
from sklearn.metrics import roc_auc_score

from settings import path, RANDOM_SEED
from load import load, write
from a2 import get_model, MyDataGenerator, get_n_fit_model

OPTUNA_STRUCTURE_STUDY_FILE = path['data']['approach_2'] / 'optuna_structure.pickle'
N_JOBS = 1

OPTUNA_TEST_POINTS = 500
OPTUNA_RANDOM_INDEXES = []
OPTUNA_BS_POINTS = 100
OPTUNA_BS_POINTS_SEED = []
OPTUNA_BS_SIZE = 1000
RS = np.random.RandomState(RANDOM_SEED)

DATA_GENERATOR = None
DATA_GENERATOR = MyDataGenerator(model_batch_size=20,
                                 pca_var=0.9,
                                 pca_sample_size=50000,
                                 force_pca_fitting=True,
                                 verbose=0)


def test_model(model, data_generator, rs=RS,
               optuna_random_indexes=OPTUNA_RANDOM_INDEXES,
               optuna_bs_points_seed=OPTUNA_BS_POINTS_SEED):
    if not optuna_random_indexes:
        optuna_random_indexes += list(rs.permutation(range(len(data_generator))))[:OPTUNA_TEST_POINTS]
    if not optuna_bs_points_seed:
        optuna_bs_points_seed += [rs.randint(1e9) for _ in range(OPTUNA_BS_POINTS)]
    
    res = []
    yt = []
    for i in optuna_random_indexes:
        x, y = data_generator[i]
        res += list(np.reshape(model.predict(x, verbose=0), (-1, )))
        yt += list(y.numpy())

    ras = []
    df = pd.DataFrame([yt, res]).T
    for _, seed in zip(range(OPTUNA_BS_POINTS), optuna_bs_points_seed):
        temp = df.sample(OPTUNA_BS_SIZE, replace=True, random_state=seed)
        ras.append(roc_auc_score(temp[0], temp[1]))
    return np.mean(ras)
    

def model_structure_objective(trial, data_generator=DATA_GENERATOR):
    param_data_generator = dict()
    if not data_generator:
        param_data_generator['model_batch_size'] = trial.suggest_int('model_batch_size', 5, 50, step=5)
        param_data_generator['pca_var'] = trial.suggest_float('pca_var', 0.9, 0.99, step=0.01)
        param_data_generator['force_pca_fitting'] = True
        param_data_generator['verbose'] = 100
    
    param_model = dict()
    param_model['multihead_count'] = trial.suggest_int('multihead_count', 1, 3, step=1)
    param_model['heads'] = trial.suggest_int('heads', 6, 20, step=2)
    param_model['ff_count'] = trial.suggest_int('ff_count', 2, 4, step=1)
    param_model['ff_wide'] = trial.suggest_int('ff_wide', 50, 950, step=100)
    param_model['ff_activation'] = trial.suggest_categorical('ff_activation', ['relu', 'elu'])
    param_model['last_ff_count'] = trial.suggest_int('last_ff_count', 2, 4, step=1)
    param_model['last_ff_wide'] = trial.suggest_int('last_ff_wide', 50, 950, step=100)
    param_model['last_ff_activation'] = trial.suggest_categorical('last_ff_activation', ['relu', 'elu'])
    param_model['dropout'] = 0.1
    
    param_model_fit = dict()
    param_model_fit['steps_per_epoch'] = 1000
    param_model_fit['epochs'] = 20
    param_model_fit['verbose'] = 2
    
    params = dict()
    params['data_generator_kw_args'] = param_data_generator
    params['model_kwargs'] = param_model
    params['model_fit_kwargs'] = param_model_fit
    if data_generator:
        params['data_generator'] = data_generator
    
    model, history, data_generator = get_n_fit_model(**params)
    return test_model(model, data_generator)


def define_model_structure(model_structure_objective=model_structure_objective, n_trials=1, n_jobs=N_JOBS):
    study = load(OPTUNA_STRUCTURE_STUDY_FILE,
                 optuna.create_study(sampler=TPESampler(), direction='maximize'))
    for _ in range(n_trials):
        study.optimize(model_structure_objective, n_trials=1, n_jobs=n_jobs)
        write(OPTUNA_STRUCTURE_STUDY_FILE, study)
    return study

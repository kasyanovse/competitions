import pickle
from pathlib import Path

import numpy as np
from numpy.random import RandomState
import pandas as pd
from sklearn.metrics import roc_auc_score, make_scorer


TEST_DATA = 'test.csv'
TRAIN_DATA = 'train.csv'

RANDOM_SEED = 1
RANDOM_GENERATOR = RandomState(RANDOM_SEED)
GET_RANDINT = lambda : RANDOM_GENERATOR.randint(0, 1e9)
TEST_SIZE = [4.5 / 7]
TARGET_FEATURE = 'cardio'
FIG_SIZES = {'small': (10, 2), 'normal': (12, 6)}
SCORER = make_scorer(roc_auc_score, needs_proba=True)

OPTUNA_STUDY_NAME = lambda x: Path(f"opt_study_{x}.pickle")

DFS_NAME = r'dfs.pickle'
FEATURE_NAMES = r'features.pickle'


def write(file_name, data):
    with open(file_name, 'wb') as handle:
        pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)

        
def read(file_name):
    with open(file_name, 'rb') as handle:
        return pickle.load(handle)

    
def split(df):
    return df.drop(columns=[TARGET_FEATURE]), df[TARGET_FEATURE]


def score(model, df):
    return SCORER(model, *split(df))


def test_bt(model, df, n_sample=500, n_tests=50, seed_generator=GET_RANDINT):
    return [score(model, df.sample(n_sample, random_state=seed_generator(),
                                   replace=True if df.shape[0] < n_sample else False))
            for _ in range(n_tests)]


def predict(model, name='', df=None, save=False):
    if df is None:
        df = read(DFS_NAME)[-1]
    t = model.predict_proba(df)[:, 1]
    t = pd.Series(t, index=df.index, name=TARGET_FEATURE)
    if save:
        t.to_csv(f'submission{name}.csv')
    else:
        return t


def get_pars_from_tune_res(name, param_filter=None, edge=0.75, par_edge=0.2, max_num_of_params=1, **kwargs):
    if param_filter is None:
        param_filter = lambda x: x
    s = read(OPTUNA_STUDY_NAME(name))
    v = sorted([(x.value, x.params) for x in s.trials if x.value and x.value > edge], key=lambda x: x[0], reverse=True)
    pars = []
    for iv in [x[1] for x in v]:
        for ip in pars:
            keys = np.intersect1d(list(iv.keys()), list(ip.keys()))
            if sum([iv[k] == ip[k] for k in keys]) / len(keys) > par_edge:
                break
        else:
            pars.append(iv)
    return list(map(param_filter, pars[:max_num_of_params]))
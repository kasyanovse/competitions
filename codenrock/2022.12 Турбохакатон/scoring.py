""" funs for testing and scoring 
    approach consist in bootstrap over testing dataframe """

import numpy as np
from scipy.stats import mvsdist, hmean
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, f1_score, rand_score, mutual_info_score, completeness_score, fowlkes_mallows_score, homogeneity_score, v_measure_score, make_scorer

from utils import TARGET_FEATURE, FIGSIZE, DATAFILE_NAMES, RANDOM_SEED, RANDOM_GENERATOR_INT, load, train_split, target_split


TEST_SIZE = 200
TEST_COUNT = 100


CLASSIFICATION_SCORERS = {'f1': make_scorer(f1_score, greater_is_better=True, needs_proba=False, average='macro'),
                          'roc_auc': make_scorer(roc_auc_score, greater_is_better=True, needs_proba=True, multi_class='ovo')}
CLUSTERIZATION_SCORERS_OLD = {'ri': make_scorer(rand_score, greater_is_better=True),
                          'mis': make_scorer(mutual_info_score, greater_is_better=True),
                          'cs': make_scorer(completeness_score, greater_is_better=True),
                          'fms': make_scorer(fowlkes_mallows_score, greater_is_better=True),
                          'hs': make_scorer(homogeneity_score, greater_is_better=True),
                          'vm': make_scorer(v_measure_score, greater_is_better=True),}
CLUSTERIZATION_SCORERS = {'cs': make_scorer(completeness_score, greater_is_better=True),
                          'hs': make_scorer(homogeneity_score, greater_is_better=True)}


""" for classification """
def sample(df, test_size, random_state):
    # return df.sample(n=test_size, replace=True, random_state=random_state)
    return train_test_split(df, train_size=test_size/df.shape[0], random_state=random_state, stratify=df[TARGET_FEATURE])[0]


def process_test(test_results):
    m, v, s = mvsdist(test_results)
    return (m.mean(), m.interval(0.9)[1] - m.mean())


def classification_score(model, df, scorers_name=['all'], test_size=TEST_SIZE, test_count=TEST_COUNT):
    """ scorers - list of scorers key in CLASSIFICATION_SCORERS dict
                  'all' means all scorers """
    if test_size < df.shape[0] * 0.5:
        test_size = df.shape[0] // 2
    
    test_samples = [sample(df, test_size, RANDOM_GENERATOR_INT(2 ** 31 - 1)) for _ in range(test_count)]
    scorers = CLASSIFICATION_SCORERS if 'all' in scorers_name else {k: CLASSIFICATION_SCORERS[k] for k in scorers_name}
    res = {name: process_test([scorer(model, *target_split(df)) for df in test_samples]) for name, scorer in scorers.items()}
    return res


""" for clusterization """
def clusterization_score(model, df, scorers_name=['all'], **kwargs):
    """ scorers - list of scorers key in CLUSTERIZATION_SCORERS dict
                  'all' means all scorers """
    scorers = CLUSTERIZATION_SCORERS if 'all' in scorers_name else {k: CLUSTERIZATION_SCORERS[k] for k in scorers_name}
    res = {name: scorer(model, *target_split(df)) for name, scorer in scorers.items()}
    return res | {'m': hmean(list(res.values()))}


def clusterization_score_by_result(ttrue, tpred, scorers_name=['all'], **kwargs):
    scorers = CLUSTERIZATION_SCORERS if 'all' in scorers_name else {k: CLUSTERIZATION_SCORERS[k] for k in scorers_name}
    res = {name: scorer._score_func(ttrue, tpred) for name, scorer in scorers.items()}
    return res | {'m': hmean(list(res.values()))}



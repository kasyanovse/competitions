""" tools for model scoring """

from math import nan

import numpy as np
import pandas as pd

from sklearn.base import is_classifier, is_regressor
from sklearn.metrics import roc_auc_score, f1_score

from settings import RANDOM_SEED


def split_age_to_bins(data):
    res = np.zeros(data.shape)
    for age in [19, 26, 36, 46, 56, 66]:
        res += (data >= age)
    res[pd.isna(data)] = nan
    return res


def score_sex(yt, yp):
    return 2 * roc_auc_score(yt, yp) - 1


def score_sex_model(model, x, y):
    return score_sex(y, model.predict_proba(x)[:, np.argmax(model.classes_ == 1)])


def score_age(yt, yp):
    return f1_score(yt, yp, average='weighted')


def score_age_model(model, x, y):
    yp = model.predict(x)
    if is_regressor(model):
        yp = split_age_to_bins(yp)
    return score_age(y, yp)


def score_both(sex, age):
    return 2 * score_age(*age) + score_sex(*sex)


def score_both_models(sex, age):
    return 2 * score_age_model(*age) + score_age_model(*sex)


def score_model(sex=None, age=None, bootstrap=True, bootstrap_steps=20, size_of_sample_for_bootstrap=10000):
    if sex is None or age is None:
        raise ValueError('sex should be (model, x, y), age should be (model, x, y)')
    sex = (sex[-1], sex[0].predict_proba(sex[1])[:, np.argmax(sex[0].classes_ == 1)])
    age = (age[-1], age[0].predict(age[1]))
    if not bootstrap:
        return score_both(sex, age)
    
    sex = pd.DataFrame(zip(*map(list, sex)), columns=[0, 1])
    age = pd.DataFrame(zip(*map(list, age)), columns=[0, 1])
    
    generator = np.random.RandomState(RANDOM_SEED)
    rand = lambda: generator.randint(1e9)
    scores = []
    for _ in range(bootstrap_steps):
        sex0 = list(map(list, sex.sample(n=size_of_sample_for_bootstrap, replace=True, random_state=rand()).values.T))
        age0 = list(map(list, age.sample(n=size_of_sample_for_bootstrap, replace=True, random_state=rand()).values.T))
        scores.append(score_both(sex0, age0))
    return np.mean(scores)

""" funs for baseline testing """

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, f1_score, make_scorer

from sklearn.linear_model import LogisticRegression
from sklearn.cluster import SpectralClustering, KMeans
from catboost import CatBoostClassifier

from utils import TARGET_FEATURE, FIGSIZE, DATAFILE_NAMES, RANDOM_SEED, load, train_split, target_split
from scoring import classification_score, clusterization_score

pd.set_option('mode.chained_assignment', None)


def cb(df):
    return CatBoostClassifier(auto_class_weights='Balanced', verbose=0, random_state=RANDOM_SEED,
                              loss_function='MultiClass', classes_count=len(df[TARGET_FEATURE].unique()))


def lr(df):
    return LogisticRegression(class_weight='balanced')


def sc(df):
    return SpectralClustering(n_clusters=len(df[TARGET_FEATURE].dropna().unique()), random_state=RANDOM_SEED)


def kmeans(df):
    return KMeans(n_clusters=len(df[TARGET_FEATURE].dropna().unique()), random_state=RANDOM_SEED)


def baseline_classification_score(df, test_size=0.5, dropna=True):
    """ eval score of baseline model over df """
    if dropna:
        df = df.dropna()
    df[TARGET_FEATURE] = df[TARGET_FEATURE].fillna(-1).astype('category').cat.codes
    df0, df1 = train_split(df, test_size=test_size)
    model = cb(df)
    model.fit(*target_split(df0))
    return classification_score(model, df1)


def baseline_clusterization_score(df, dropna=True):
    """ eval score of baseline model over df """
    if dropna:
        df = df.dropna()
    df[TARGET_FEATURE] = df[TARGET_FEATURE].fillna(-1).astype('category').cat.codes
    model = kmeans(df)
    model.fit(*target_split(df))
    return clusterization_score(model, df)
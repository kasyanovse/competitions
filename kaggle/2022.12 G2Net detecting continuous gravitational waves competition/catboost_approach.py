# -*- coding: utf-8 -*-
from math import ceil
from pathlib import Path

import numpy as np
from scipy.stats import skew, kurtosis
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm

from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
from catboost import CatBoostClassifier, CatBoostRegressor

from settings import PATH, SETTINGS
from utils import *
from data_convert import process_data_approach_2, MAX_VAL
from scoring import ClassificationScoreInterface, RegressorScoreInterface

PATH_TO_DATA = PATH['data']['amp_2']['generated']
DUMP_FILE = PATH['catboost_db_file']
TARGET_FEATURE = SETTINGS['catboost']['target_feature']
EDGE = SETTINGS['edge_amp_2']

def split(df):
    return df.drop(columns=TARGET_FEATURE), df[TARGET_FEATURE]


def split_to_test_train(df, train_size=0.8):
    if len(df[TARGET_FEATURE].unique()) > 3:
        return train_test_split(df, train_size=train_size, random_state=SETTINGS['random_seed'])
    else:
        return train_test_split(df, train_size=train_size, stratify=df[TARGET_FEATURE], random_state=SETTINGS['random_seed'])


class BaseCB():
    def __init__(self, edge, tqdm=tqdm,
                 verbose=0, random_state=SETTINGS['random_seed'],
                 predict_mode=None, drop_0=False, **kwargs):
        super().__init__(verbose=verbose,
                         random_state=random_state, **kwargs)
        self.my_tqdm = tqdm
        self.my_edge = edge
        self.my_df = None
        self.drop_0 = drop_0
        self.predict_mode = predict_mode
        self.funs = create_funs()

    def predict(self, x, y=None):
        if self.predict_mode == 'with_preprocessing':
            x = self.preprocesser(x)
        elif self.predict_mode == 'with_full_preprocessing':
            x = self.full_preprocesser(x)
        return super().predict(x)
    
    def predict_proba(self, x, y=None):
        if self.predict_mode == 'with_preprocessing':
            x = self.preprocesser(x)
        elif self.predict_mode == 'with_full_preprocessing':
            x = self.full_preprocesser(x)
        return super().predict_proba(x)
    
    def preprocesser(self, a):
        max_val, max_val2 = 1 / MAX_VAL, 1 / (MAX_VAL ** 2)
        return {i + 1: fun(a, max_val, max_val2) for i, fun in enumerate(self.funs)}
    
    def full_preprocesser(self, x):
        x = self.preprocesser(x)
        # for col in x:
        #     x[col] = (x[col] - self.my_min[col]) / (self.my_max[col] - self.my_min[col])
        x = np.array(list(x.values()))
        x = np.reshape(x, (1, -1))
        # x = self.my_pca.transform(x)
        return x
    
    def read_data(self, force=False, path=PATH_TO_DATA, dump_file=DUMP_FILE):
        if self.my_df is None or force:
            dump_file = Path(dump_file)
            if dump_file.exists() and not force:
                df = read(dump_file)
            else:
                files = list(path.glob('*' + SETTINGS['base_format_for_amp_2']))
                labels = pd.read_csv(path / SETTINGS['labels_file_name'], index_col=0)
                snr = pd.read_csv(path / SETTINGS['snr_file_name'], index_col=0)
                df = []
                for file in self.my_tqdm(files):
                    a = np.array(Image.open(file)) if SETTINGS['base_format_for_amp_2'] == '.png' else np.load(file)
                    name = file.stem
                    df.append({'id': name}
                              | self.preprocesser(a)
                              | {TARGET_FEATURE: snr.loc[name][TARGET_FEATURE]})
                df = pd.DataFrame(df).set_index('id')
                write(dump_file, df)
            if self.my_edge is not None:
                df[TARGET_FEATURE] = (df[TARGET_FEATURE] > self.my_edge).astype(int)
            if self.drop_0:
                df = df[df[TARGET_FEATURE] != 0]
            self.my_df = df
        self.prepare_data()
        return self.my_df
    
    def prepare_data(self):
        if self.my_df is None:
            raise ValueError('there are no data is saved in model')
        
        df = self.my_df
        x, y = split(df)
        self.my_min = dict()
        self.my_max = dict()
        # for col in x:
        #     self.my_min[col] = x[col].min()
        #     self.my_max[col] = x[col].max()
        #     x[col] = (x[col] - x[col].min()) / (x[col].max() - x[col].min())
        df = pd.concat([x, y], axis=1)
        self.my_pca = PCA(0.999)
        # x = pd.DataFrame(self.my_pca.fit_transform(x), index=self.my_df.index)
        x = pd.DataFrame(x, index=self.my_df.index)
        self.my_data = (x, y)
        
        df1, df2 = split_to_test_train(df)
        x1, y1 = split(df1)
        x2, y2 = split(df2)
        
        pca = PCA(0.999)
        pca.fit(x1)
        # x1 = pca.transform(x1)
        # x2 = pca.transform(x2)
        x1 = pd.DataFrame(x1, index=df1.index)
        x2 = pd.DataFrame(x2, index=df2.index)
        
        self.my_train = (x1, y1)
        self.my_test = (x2, y2)


class MyCB(BaseCB, ClassificationScoreInterface, CatBoostClassifier):
    def __init__(self, edge=EDGE, **kwargs):
        super().__init__(edge=edge, objective='Logloss', boosting_type='Ordered',
                         bootstrap_type='Bernoulli', **kwargs)


class MyCBR(BaseCB, RegressorScoreInterface, CatBoostRegressor):
    def __init__(self, drop_0=True, **kwargs):
        super().__init__(edge=None, drop_0=drop_0, **kwargs)


def create_funs():
    def var(array, max_val, max_val2):
        return np.var(np.reshape(array, (-1, ))) * max_val2

    def max_var_by_row(array, max_val, max_val2):
        return np.max(np.var(array, axis=1)) * max_val2

    def max_mean_by_row(array, max_val, max_val2):
        return np.max(np.mean(array, axis=1)) * max_val

    def var_of_var_by_row(array, max_val, max_val2):
        return np.var(np.var(array, axis=1)) * max_val2 * max_val

    def mean_median_delta(array, max_val, max_val2):
        array = np.reshape(array, (-1, ))
        return (np.mean(array) - np.median(array)) * max_val

    def max_row_mean_delta_from_common_mean(array, max_val, max_val2):
        means = np.mean(array, axis=1)
        return np.max(means - np.mean(means)) * max_val

    def min_diff_by_time(array, max_val, max_val2):
        return np.abs(np.min(np.reshape(np.diff(array, axis=1), (-1, )))) * max_val

    def mean_diff_by_time(array, max_val, max_val2):
        return np.abs(np.mean(np.reshape(np.diff(array, axis=1), (-1, )))) * max_val

    def var_diff_by_time(array, max_val, max_val2):
        return np.abs(np.var(np.reshape(np.diff(array, axis=1), (-1, )))) * max_val2
    
    def my_skew(array, max_val, max_val2):
        return skew(np.reshape(array, (-1, )))
    
    def my_kurtosis(array, max_val, max_val2):
        return kurtosis(np.reshape(array, (-1, )))
    
    # origin_funs = [var, max_var_by_row, max_mean_by_row, var_of_var_by_row, mean_median_delta,
    #                max_row_mean_delta_from_common_mean, min_diff_by_time, mean_diff_by_time, var_diff_by_time]
    # origin_funs = [var, max_mean_by_row, mean_median_delta, max_row_mean_delta_from_common_mean, min_diff_by_time, mean_diff_by_time]
    # origin_funs = [var, max_mean_by_row, mean_diff_by_time]
    # origin_funs = [max_mean_by_row, mean_diff_by_time]
    # origin_funs = [var, mean_diff_by_time]
    # origin_funs = [var, min_diff_by_time]
    # origin_funs = [var, max_row_mean_delta_from_common_mean, mean_diff_by_time]
    # origin_funs = [var_diff_by_time, min_diff_by_time, mean_diff_by_time]
    # origin_funs = [my_skew, my_kurtosis, mean_diff_by_time]
    origin_funs = [my_skew, my_kurtosis]

    # то же самое, но удалить часть строк с минимальным параметром
    def _del_half_of_minimal_means_row(array):
        length = int(array.shape[1] * 0.2)
        idx = np.argsort(np.mean(array, axis=1))[:length]
        return array[idx, :]

    def _del_half_of_minimal_var_row(array):
        length = int(array.shape[1] * 0.2)
        idx = np.argsort(np.var(array, axis=1))[:length]
        return array[idx, :]

    # то же самое, но применим разные функции
    def _power4(array):
        return array ** 4

    def _power01(array):
        return array ** 0.1

    def _exp(array):
        return np.exp(array)

    def _log(array):
        return np.log(array + 1e-3)
    
    
    # нормализация
    def _normalize(array):
        return (array - np.mean(array)) / np.std(array)
    
    def _normalize2(array):
        return (array - np.mean(array)) / np.sum(array ** 2)
    
    # prep_funs = [lambda x: x, _del_half_of_minimal_means_row, _del_half_of_minimal_var_row, _power4, _power01]
    # prep_funs = [lambda x: x, _del_half_of_minimal_means_row]
    # prep_funs = [lambda x: x]
    # prep_funs = [_del_half_of_minimal_means_row]
    prep_funs = [_normalize]
    # prep_funs = [_normalize2]
    
    funs = []
    for prep_fun in prep_funs:
        for origin_fun in origin_funs:
            funs.append(lambda x, *args, origin_fun=origin_fun, prep_fun=prep_fun: origin_fun(prep_fun(x), *args))
    return funs
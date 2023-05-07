""" tools for 2 approach """

import time

import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from settings import path, RANDOM_SEED
from load import load, write, load_col, load_target

from process_data import (dataframe_group_to_dict_by_user, simple_process_category,
                          simple_process, process_url_n_set_to_data,
                          get_new_features_from_date, get_new_features_from_part_of_day,
                          process_manufacturer, dataframe_group_indexes_to_dict_by_user)


DATA_PATH = path['data']['approach_3']
COMMON_FEATURE_PATH = DATA_PATH / 'common_features.pickle'
COMMON_FEATURES_DICT_PATH = DATA_PATH / 'common_features_dict.pickle'
PCA_FILE_PATH = DATA_PATH / 'data_generator_pca.pickle'


def get_common_feature():
    df = load(COMMON_FEATURE_PATH)
    if df is None:
        features= [simple_process(load_col('user_id')),
                   process_url_n_set_to_data(load_col('url_host'), replace_by_rare_deep={1: 10, 2: 10}),
                   process_manufacturer(load_col('cpe_manufacturer_name')),
                   simple_process_category(load_col('region_name')),
                   simple_process_category(load_col('part_of_day')),
                   simple_process(load_col('request_cnt')),
                   simple_process(load_col('price')),
                   get_new_features_from_date(load_col('date')),
                   ]
        df = pd.concat([features.pop() for _ in range(len(features))], axis=1)
        df = df.drop(columns=['gov', 'ucoz'])
        if 'price' in df:
            # df['price'] /= 50000
            df['price'] = (df['price'] - df['price'].min()) / (df['price'].max() - df['price'].min())
            df['price'] =  df['price'] * 0.5 / df['price'].mean()
        if 'request_cnt' in df:
            df['request_cnt'] /= (df['request_cnt'].mean() / 0.8)
        df = df.sort_values('user_id').reset_index(drop=True)
        write(COMMON_FEATURE_PATH, df)
    return df

def get_common_feature_dict(**kwargs):
    return dataframe_group_indexes_to_dict_by_user(COMMON_FEATURES_DICT_PATH, get_common_feature, **kwargs)


def PrintPasTriangle(rows, start_val=1):
    row = [start_val]
    for i in range(rows):
        row = [sum(x) for x in zip([0]+row, row+[0])]
    return row


def get_model(data_shape,
              ff_count=3, ff_activation='relu', ff_wide=None,
              ff_h_count=1, ff_h_activation='elu', ff_h_wide=None,
              dropout_h=0.5, dropout_v=0.1,
              random_seed=RANDOM_SEED):
    if ff_wide is None:
        ff_wide = data_shape[0]
    if ff_h_wide is None:
        ff_h_wide = data_shape[0]
    if dropout_v is not None:
        ff_wide = int(ff_wide / (1 - dropout_v))
    if dropout_h is not None:
        ff_h_wide = int(ff_h_wide / (1 - dropout_h))
    
    inputs = keras.Input(shape=list(data_shape))
    x = inputs
    # x = layers.LayerNormalization(epsilon=1e-6)(x)
    # x = layers.BatchNormalization()(x)
    x = layers.Dense(ff_wide)(x)
    
    for _ in range(ff_count):
        x0 = x
        for _ in range(ff_h_count):
            x = layers.Dense(ff_h_wide, activation=ff_h_activation)(x)
            if dropout_h is not None:
                x = layers.Dropout(dropout_h, seed=random_seed)(x)
        x = layers.Dense(ff_wide, activation=ff_activation)(x)
        if dropout_v is not None:
            x = layers.Dropout(dropout_v, seed=random_seed)(x)
        x = layers.LayerNormalization(epsilon=1e-6)(x)
        x = x + x0
    
    outputs = layers.Dense(1, activation='sigmoid')(x)
    model = keras.Model(inputs, outputs)
    model.compile(loss=tf.keras.losses.BinaryCrossentropy(),
                  optimizer=keras.optimizers.Adam(learning_rate=1e-4),
                  metrics=[tf.keras.metrics.BinaryAccuracy(), tf.keras.metrics.AUC()])
    return model


class MyDataGenerator(tf.keras.utils.Sequence):
    def __init__(self,
                 target_col='is_male',
                 train_share=1, target_type='train',
                 pca_var=0.99, pca_sample_size=20000,
                 force_pca_fitting=False,
                 random_seed=RANDOM_SEED):
        self.target_col = target_col
        self.random_seed = random_seed
        
        self.df = get_common_feature()
        self.dfis = get_common_feature_dict(df=self.df)
        self.df = self.df.drop(columns=['user_id'])
        self.dfis.df = self.df
        self.target = load_target()[self.target_col].dropna()
        if train_share < 1:
            t1, t2 = train_test_split(self.target, train_size=train_share,
                                      random_state=self.random_seed, stratify=self.target)
            self.target = t1 if target_type == 'train' else t2
        
        self.cols = None
        self.noncols = None
        self.pca_file_name = PCA_FILE_PATH
        self.pca = None
        self.prepare_pca(pca_var, pca_sample_size, force_pca_fitting=force_pca_fitting)
    
    def __getitem__(self, index):
        x, y = self.get_user(self.target.index[index])
        return tf.convert_to_tensor(x), tf.convert_to_tensor(y)

    def __len__(self):
        return self.target.shape[0]
    
    def get_user(self, user_id):
        x = self.dfis[user_id]
        if self.cols is None:
            self.cols = [col for col in x if x[col].dtype == 'category' and len(x[col].cat.categories) > 2]
            self.noncols = [col for col in x if col not in self.cols]
        
        x_cols = pd.get_dummies(x[self.cols], dummy_na=True)
        x_noncols = x[self.noncols]
        x = pd.concat([x_noncols.astype(float).mean(), x_cols.mean()])
        
        x = np.reshape(x.values, (1, -1))
        y = np.array([self.target.loc[user_id]])
        
        if self.pca is not None:
            x = self.pca.transform(x)
        return x, y
    
    def prepare_pca(self, pca_var, pca_sample_size, force_pca_fitting=False):
        self.pca = load(self.pca_file_name)
        if self.pca is None or force_pca_fitting:
            sample = self.target.sample(n=pca_sample_size, random_state=self.random_seed)
            sample = np.concatenate([self.get_user(user_id)[0] for user_id in sample.index])
            self.pca = PCA(pca_var, random_state=self.random_seed).fit(sample)
            write(self.pca_file_name, self.pca)
        self.data_shape = (self.pca.n_components_, )

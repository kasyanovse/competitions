""" tools for 2 approach """

import time

import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.decomposition import PCA
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


DATA_PATH = path['data']['approach_2']
COMMON_FEATURE_PATH = DATA_PATH / 'common_features.pickle'
COMMON_FEATURES_DICT_PATH = DATA_PATH / 'common_features_dict.pickle'


def get_common_feature():
    df = load(COMMON_FEATURE_PATH)
    if df is None:
        features= [simple_process(load_col('user_id')),
                   process_url_n_set_to_data(load_col('url_host'), replace_by_rare_deep={1: 10, 2: 10}),
                   process_manufacturer(load_col('cpe_manufacturer_name')),
                   simple_process_category(load_col('region_name')),
                   simple_process_category(load_col('part_of_day')),
                   simple_process(load_col('request_cnt')),
                   # simple_process_category(load_col('cpe_type_cd')),
                   simple_process(load_col('price')),
                   get_new_features_from_date(load_col('date')),
                   ]
        df = pd.concat([features.pop() for _ in range(len(features))], axis=1)
        df = df.drop(columns=['gov', 'ucoz'])
        if 'price' in df:
            # df['price'] = df['price'] / 50000
            df['price'] = (df['price'] - df['price'].max()) / (df['price'].max() - df['price'].min()) * 0.5 / df['price'].mean()
        df = df.sort_values('user_id').reset_index(drop=True)
        write(COMMON_FEATURE_PATH, df)
    return df

def get_common_feature_dict():
    # return dataframe_group_to_dict_by_user(COMMON_FEATURES_DICT_PATH, get_common_feature)
    return dataframe_group_indexes_to_dict_by_user(COMMON_FEATURES_DICT_PATH, get_common_feature)


def get_model(data_shape,
              multihead_count=2, heads=10,
              ff_count=2, ff_wide=100, ff_activation='relu',
              last_ff_count=2, last_ff_wide=100, last_ff_activation='relu',
              filters=16,
              dropout=0.1):
    inputs = keras.Input(shape=data_shape)
    x = inputs
    for _ in range(multihead_count):
        x0 = x
        # x = layers.LayerNormalization(epsilon=1e-6)(x)
        x = layers.MultiHeadAttention(key_dim=data_shape[1],
                                      num_heads=heads,
                                      dropout=dropout)(x, x)
        x = layers.LayerNormalization(epsilon=1e-6)(x)
        x = x + x0
        x0 = x
        for _ in range(ff_count):
            x = layers.Dense(ff_wide, activation=ff_activation)(x)
        x = layers.Dropout(dropout)(x)
        x = layers.LayerNormalization(epsilon=1e-6)(x)
        x = layers.Dense(data_shape[1], activation=ff_activation)(x)
        x = x + x0

    # x = layers.GlobalAveragePooling1D()(x)
    x = layers.GlobalMaxPooling1D()(x)
    
    # x = layers.Reshape(list(data_shape) + [1])(x)
    # reshape = data_shape[0]
    # while reshape > 1:
    #     if reshape % 2 == 0:
    #         s = 2
    #     elif reshape % 3 == 0:
    #         s = 3
    #     elif reshape % 5 == 0:
    #         s = 5
    #     else:
    #         raise ValueError()
    #     x = layers.Conv2D(filters=filters, kernel_size=(s, 1), strides=(s, 1))(x)
    #     reshape /= s
    #     filters *= 2
    # x = layers.Flatten()(x)
    
    for _ in range(last_ff_count):
        x = layers.Dense(last_ff_wide, activation=last_ff_activation)(x)
    outputs = layers.Dense(1, activation='sigmoid')(x)
    model = keras.Model(inputs, outputs)
    model.compile(loss=tf.keras.losses.BinaryCrossentropy(),
                  optimizer=keras.optimizers.Adam(learning_rate=1e-4),
                  metrics=[tf.keras.metrics.BinaryAccuracy()])
    return model


def get_n_fit_model(data_generator=None,
                    data_generator_kw_args=dict(),
                    model_kwargs=dict(),
                    model_fit_kwargs=dict(),
                    get_model_fun=get_model,
                    plot=False):
    if data_generator is None:
        data_generator = MyDataGenerator(**data_generator_kw_args)
    model = get_model_fun(data_generator.data_shape, **model_kwargs)
    history = model.fit(data_generator, **model_fit_kwargs)
    if plot:
        data = history.history['binary_accuracy']
        _, ax = plt.subplots(figsize=(20, 3))
        ax.plot(data)
        ax.plot(pd.Series(data).rolling(5).mean())
        ax.set_title(f"model accuracy {np.mean(data):.3f}")
        ax.set_ylabel('accuracy')
        ax.set_xlabel('epoch')
        plt.show()
    return model, history, data_generator


class MyDataGenerator(tf.keras.utils.Sequence):
    def __init__(self, target_col='is_male',
                 model_batch_size=10,
                 pca_var=0.99, pca_sample_size=20000,
                 force_pca_fitting=False,
                 random_seed=RANDOM_SEED,
                 verbose=0):
        self.model_batch_size = model_batch_size
        self.target_col = target_col
        self.random_seed = random_seed
        self.verbose = verbose
        
        self.load_df()
        self.load_target()
        self.prepare_data_processor(pca_var, pca_sample_size, force_pca_fitting=force_pca_fitting)
        
        self.df = self.df.drop(columns=['user_id'])
        self.load_dfis()
        
        a = self.target.index
        step = 20
        self.indexes = {num: a[i * step:(i + 1) * step] for num, i in enumerate(range(len(a) // step))}
        self.elements_count = len(self.indexes)
    
    def __getitem__(self, index):
        data = [self.get_user(user_id) for user_id in self.indexes[index]]
        x = tf.convert_to_tensor(np.concatenate([x[0] for x in data], 0))
        y = tf.convert_to_tensor(np.concatenate([x[1] for x in data], 0))
        return x, y
    
    
    def get_user(self, user_id):
        x = self.data_processor(self.dfis[user_id])
        x = x.drop_duplicates()
        res = x.shape[0] % self.model_batch_size
        if res != 0:
            x = pd.concat([x, x.sample(self.model_batch_size - res, replace=True)])
        x = np.reshape(x.values, (-1, self.model_batch_size, x.shape[1]))
        y = np.zeros(x.shape[0]) + self.target.loc[user_id]
        # return tf.convert_to_tensor(x), tf.convert_to_tensor(y)
        return x, y


    def __len__(self):
        return self.elements_count
    
    def load_dfis(self):
        if not hasattr(self, 'dfis') or self.dfis is None:
            if self.verbose > 50: print('load main dataframe in dict splitted by users', end=' ')
            self.dfis = get_common_feature_dict()
            self.dfis.df = self.df
            if self.verbose > 50: print(f"(loaded)")
    
    def load_df(self):
        if not hasattr(self, 'df') or self.df is None:
            if self.verbose > 50: print('load main dataframe', end=' ')
            self.df = get_common_feature()
            if self.verbose > 50: print(f"({self.df.memory_usage().sum() / 2 ** 30:.1f} Gb)")
    
    def load_target(self):
        if self.target_col is None:
            self.target = None
        if not hasattr(self, 'target') or self.target is None:
            if self.verbose > 50: print('load target dataframe', end=' ')
            # self.target = get_common_target()
            self.target = load_target().drop(columns=['user_id'])[self.target_col].dropna().sort_index()
            if self.verbose > 50: print(f"({self.target.memory_usage() / 2 ** 20:.0f} Mb)")
    
    def prepare_data_processor(self, pca_var, pca_sample_size, force_pca_fitting=False):
        """ data_processor - fun (saved in property) that process data before return as batch """
        pca_file_name = path['data']['approach_2'] / 'pca_model.pickle'
        self.pca = load(pca_file_name)
        if self.pca is None or force_pca_fitting:
            self.load_df()
            if self.verbose > 50: print('prepare pca model', end=' ')
            sample = self.df.drop(columns=['user_id']).sample(n=pca_sample_size, random_state=self.random_seed)
            sample = pd.get_dummies(sample, dummy_na=True)
            if self.verbose > 50: print(f"({sample.shape[0]} samples, {sample.memory_usage().sum() / 2 ** 20:.0f} Mb)")
            if self.verbose > 50: print('fit pca model', end=' ')
            self.pca = PCA(pca_var, random_state=self.random_seed).fit(sample)
            if self.verbose > 50: print('(fitted)')
            write(pca_file_name, self.pca)
        self.data_shape = (self.model_batch_size, self.pca.n_components_)

        def data_processor(x, pca=self.pca):
            ind = x.index
            x = pd.get_dummies(x, dummy_na=True)
            return pd.DataFrame(pca.transform(x), ind)
        self.data_processor = data_processor
        
#     def prepare_data_processor(self, *args, **kwargs):
#         def data_processor(x):
#             ind = index=x.index
#             x = pd.get_dummies(x, dummy_na=True)
#             return pd.DataFrame(x, ind)
#         self.data_processor = data_processor
        
#         temp = self.data_processor(self.df.iloc[:2].drop(columns=['user_id']))
#         self.data_shape = (self.model_batch_size, temp.shape[1])


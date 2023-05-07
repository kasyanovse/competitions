""" utils for data loading """

from math import nan
import pickle

import pandas as pd
from pandas.api.types import union_categoricals
import pyarrow
import pyarrow.parquet as pq

from settings import path
from score import split_age_to_bins


def load(file_name, return_if_file_does_not_exist=None):
    if file_name.exists():
        with open(file_name, 'rb') as f:
            return pickle.load(f)
    return return_if_file_does_not_exist


def write(file_name, data):
    with open(file_name, 'wb') as f:
        pickle.dump(data, f)


def load_splitted_data():
    """ splitted data - data splitted by known target
        look for splitted data in files
        otherwise prepare and save it in files """
    if any(not x.exists() for x in (path['data']['splitted_files_with_target_data'],
                                    path['data']['splitted_files_without_target_data'])):
        print('splitted data is not found')
        target = load_target()
        print('target is loaded')
        
        data = load_data()
        print('data is loaded') 
        switcher = data['user_id'].isin(target['user_id'])
        data_wot = data[~switcher]
        write(path['data']['splitted_files_with_target_data'], data_wot)
        data_wt = data[switcher]
        write(path['data']['splitted_files_without_target_data'], data_wt)
        del data
    else:
        data_wot = load(path['data']['splitted_files_with_target_data'])
        data_wt = load(path['data']['splitted_files_without_target_data'])
        
    return (data_wot, None), (data_wt, target)


def split_by_target(data, target):
    return data[~data['user_id'].isin(target['user_id'])], data[data['user_id'].isin(target['user_id'])]


def load_data():
    """ try to load processed data if exists
        else load origin data, process it, save and return it """
    if path['data']['processed_data_file'].exists():
        return load(path['data']['processed_data_file'])
    else:
        print('there is no processed file\nstart data processing')
        category_cols = ['region_name', 'city_name', 'cpe_manufacturer_name',
                         'cpe_model_name', 'cpe_type_cd', 'cpe_model_os_type', 
                         'part_of_day', 'url_host', 'date']
        cols_to_type = {col: 'category' for col in category_cols}
        cols_to_type |= {'user_id': 'int32',
                         'price': 'float32',
                         'request_cnt': 'int8'}


        datas = []
        for data_file in path['data']['origin_data'].glob('*.parquet'):
            data = pd.read_parquet(data_file, engine='pyarrow')
            for col, type_ in cols_to_type.items():
                data[col] = data[col].astype(type_)
            datas.append(data)
        
        cats = {col: union_categoricals([data[col] for data in datas]).categories for col in category_cols}
        for col in category_cols:
            for data in datas:
                data[col] = pd.Categorical(data[col], categories=cats[col])
        data = pd.concat(datas, ignore_index=True)
        del datas
        
        # fill empty price
        for col in ('cpe_model_name', 'cpe_manufacturer_name'):
            models = data[data['price'].isna()][col].unique()
            prices = {m: data[data[col] == m]['price'].dropna().median() for m in models}
            data['price'] = data['price'].where(~data['price'].isna(), data[col].apply(lambda x, prices=prices: prices.get(x, nan)))
        data['price'] = data['price'].where(~data['price'].isna(), data['price'].median())
        data['price'] = data['price'].astype('float32')

        write(path['data']['processed_data_file'], data)
        return data


def load_col(col, index_to_return=None):
    """ try to load one col of data from file if exists """
    file_name = path['data']['processed_data_file']
    file_name = file_name.with_stem(file_name.stem + f'_{col}')
    if file_name.exists():
        data_col = load(file_name)
    else:
        data_col = load_data()[col]
        write(file_name, data_col)
    
    if index_to_return is not None:
        data_col = data_col.loc[index_to_return]
    return data_col


def load_target():
    """ try to load processed target if exists
        else load origin target, process it, save and return it """
    if path['data']['processed_data_file_target'].exists():
        return load(path['data']['processed_data_file_target'])
    else:
        data = pd.read_parquet(path['data']['origin_data_file_target'], engine='pyarrow')
        
        data['is_male'] = data['is_male'].replace({'1': 1, '0': 0, 'NA': nan}).astype('float32').astype('category')

        # age to bins
        data['age_bin'] = split_age_to_bins(data['age'])
        data['age_bin'] = data['age_bin'].astype('float32').astype('category')

        # process age
        data['age'] = data['age'].astype('float32')
        # data = data.drop(columns='age')
        
        write(path['data']['processed_data_file_target'], data)
        return data

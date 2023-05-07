import numpy as np
import scipy
import pandas as pd
import polars as pl
import implicit
from sklearn.model_selection import train_test_split

from settings import path, RANDOM_SEED
from load import load, write, load_col, load_target
from process_data import (dataframe_group_to_dict_by_user, simple_process_category,
                      simple_process, process_url_n_set_to_data,
                      get_new_features_from_date, get_new_features_from_part_of_day,
                      process_manufacturer, dataframe_group_indexes_to_dict_by_user, process_url_n_set_to_data_full)


DATA_PATH = path['data']['approach_5']
COMMON_FEATURE_PATH = DATA_PATH / 'common_features.parquet'
DATA_FILES = [COMMON_FEATURE_PATH.with_stem(COMMON_FEATURE_PATH.stem + f'_{i}') for i in range(10)]
PARQUET_FILES = {file_name:file_name.with_suffix('.parquet') for file_name in DATA_FILES}
CATEGORIES_COLS_FILE = DATA_PATH / 'categories_cols.pickle'


def preprocess_data_with_pandas_for_polars():
    print('start preparing data for polars')
    categories_cols = []
    def temp(file_name, features, categories_cols=categories_cols):
        df = pd.concat([features.pop() for _ in range(len(features))], axis=1)
        df = df.sort_values('user_id').reset_index(drop=True)
        # df = df.drop(columns='user_id')
        df.columns = [str(x) for x in df]
        for col in df.dtypes[df.dtypes == 'category'].index:
            categories_cols.append(col)
            df[col] = pd.factorize(df[col])[0]
        print('try to save', PARQUET_FILES[file_name], end=' ')
        df.to_parquet(PARQUET_FILES[file_name])
        print('all is ok')
    
    data_files = [x for x in reversed(DATA_FILES)]
    temp(data_files.pop(),
        [simple_process(load_col('user_id')),
        process_url_n_set_to_data_full(load_col('url_host'))])
    url_cols = [0, 1, 2, 3, 4]
    temp(data_files.pop(),
        [simple_process(load_col('user_id')),
        process_url_n_set_to_data(load_col('url_host'), url_cols=url_cols, replace_by_rare_deep={col: 1 for col in url_cols})])
    # temp(data_files.pop(),
    #     [simple_process(load_col('user_id')),
    #     process_manufacturer(load_col('cpe_manufacturer_name'))])
    # temp(data_files.pop(),
    #     [simple_process(load_col('user_id')),
    #     simple_process_category(load_col('region_name'))])
    # temp(data_files.pop(),
    #     [simple_process(load_col('user_id')),
    #     simple_process(load_col('price')),
    #     get_new_features_from_date(load_col('date'))])
    temp(data_files.pop(),
        [simple_process(load_col('user_id')),
        simple_process_category(load_col('part_of_day')),
        simple_process(load_col('request_cnt')),])
    write(CATEGORIES_COLS_FILE, categories_cols)
    

def get_data(lazy=False):
    if COMMON_FEATURE_PATH.exists():
        print(1)
        if lazy:
            dfs = pl.scan_parquet(COMMON_FEATURE_PATH)
        else:
            dfs = pl.read_parquet(COMMON_FEATURE_PATH)
    else:
        print(2)
        if not any(x.exists() for x in PARQUET_FILES):
            preprocess_data_with_pandas_for_polars()
        print(3)
        existing_cols = []
        dfs = None
        for file in [x for x in PARQUET_FILES.values() if x.exists()]:
            print(file)
            df = pl.read_parquet(file)
            print(' ' * 4 + '1')
            df = df.drop(columns=[x for x in df.columns if x in existing_cols])
            print(' ' * 4 + '2')
            existing_cols.extend(df.columns)
            print(' ' * 4 + '3')
            dfs = df if dfs is None else pl.concat([dfs, df], how='horizontal')
            print(' ' * 4 + '4')
            
        # processing
        print(4)
        forbidden_cols = ['user_id', 'request_cnt']
        cat_cols = load(CATEGORIES_COLS_FILE)
        for col in [x for x in dfs.columns if x not in (cat_cols + forbidden_cols)]:
            q = (dfs[col].quantile(0.1), dfs[col].quantile(0.9))
            if abs(q[1] - q[0]) > 1:
                dfs = dfs.with_columns((pl.col(col) - q[0]) / (q[1] - q[0]) - 0.5)
        dfs.write_parquet(COMMON_FEATURE_PATH)
    return dfs


def als_for_col(df0, col, drop_m1=True, factors=50, iterations=50, regularization=0.5):
    user_dict = dict(zip(df0['user_id'].unique(), range(df0.shape[0])))
    col_dict = dict(zip(df0[col].unique(), range(df0.shape[0])))
    values = df0.groupby(['user_id', col]).agg(pl.col('request_cnt').sum()).sort('user_id')
    if drop_m1:
        values = values.filter(pl.col(col) != -1)

    v = values['request_cnt'].to_numpy()
    r = values['user_id'].map_dict(user_dict).to_numpy()
    c = values[col].map_dict(col_dict).to_numpy()
    data = scipy.sparse.coo_matrix((v, (r, c)), shape=(r.max() + 1, c.max() + 1))
    als = implicit.approximate_als.NMSLibAlternatingLeastSquares(factors=factors, iterations=iterations, use_gpu=False, random_state=RANDOM_SEED,
                                                                 calculate_training_loss=False, regularization=regularization)
    als.fit(data)
    return pl.DataFrame(als.model.user_factors).with_columns(pl.Series(name='user_id', values=user_dict.values()))


def prepare_features(df, cols=('cat_1', 'cat_2', '1', '2'), **kwargs):
    res = [als_for_col(df, col, **kwargs) for col in cols]
    df = None
    for r in res:
        df = r if df is None else df.join(r, on='user_id')
        df.columns = [col if col == 'user_id' else str(i) for i, col in enumerate(df.columns)]
    return df.select([pl.col('user_id'), pl.col('*').exclude('user_id')])


def prepare_test_data(df, col, test_size=0.1):
    t = load_target()[col].dropna()
    t1, t2 = train_test_split(t, test_size=test_size, stratify=t, random_state=RANDOM_SEED)
    f1, f2 = [df.filter(pl.col('user_id').is_in(list(t.index))).join(pl.DataFrame(t.reset_index().to_numpy(), schema={'user_id': int, col: int}), on='user_id') for t in (t1, t2)]
    temp = lambda f: (f.select(pl.col('*').exclude(['user_id', col])).to_numpy(), f[col].to_numpy())
    return temp(f1), temp(f2)


def fit_n_test(model, scorer, data):
    model.fit(*data[0])
    result = {'train': scorer(model, *data[0]), 'test': scorer(model, *data[1])}
    return ((model, *data[0]), (model, *data[1])), result
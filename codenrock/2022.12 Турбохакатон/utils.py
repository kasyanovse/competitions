import pickle
from pathlib import Path

import numpy as np
from numpy.random import RandomState
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA

RANDOM_SEED = 1
RANDOM_GENERATOR = RandomState(RANDOM_SEED)
RANDOM_GENERATOR_INT = RandomState(RANDOM_SEED)
TARGET_FEATURE = 'target'
ORIGIN_TARGET_FUTURE = 'РЕЖИМ'
TIME_FEATURE = 'time'
ORIGIN_TIME_FEATURE = 'Параметр'
TRANSIENT_FEATURE = 'transient'
DATAFILE_NAMES = ('data/d1.csv', 'data/d2.csv', 'data/d3.csv')
DUMP_PATH = Path('dump')
SOLUTION_FILENAME = Path('solution/res.csv')
N_JOBS=5

FIGSIZE = (25, 6)
FIGSIZE_NARROW = (25, 3)


def read(file_name, default_value=None):
    try:
        with open(file_name, 'rb') as f:
            return pickle.load(f)
    except FileNotFoundError:
        return default_value


def write(file_name, data):
    with open(file_name, 'wb') as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)


def load(file_name, dtype='float32', try_to_read_dump=True, dropna=False):
    # try to read from dump
    pars = (dtype, )
    file_name_pickle = DUMP_PATH / (Path(file_name).stem + '.pickle')
    if try_to_read_dump and file_name_pickle.exists():
        data_pickle = read(file_name_pickle)
        if pars in data_pickle:
            return data_pickle[pars]
    else:
        data_pickle = dict()
    
    # read data from csv if needed
    df = pd.read_csv(file_name, sep=';', encoding='cp1251')
    df = df.rename(columns={ORIGIN_TIME_FEATURE: TIME_FEATURE, ORIGIN_TARGET_FUTURE: TARGET_FEATURE})
    #df = df.rename(columns={col: col[1:] for col in df.columns if col[0] == 'х'})
    df = df.dropna(axis=0, how='all')
    df = df.dropna(axis=1, how='all')
    df[TIME_FEATURE] = pd.to_datetime(df[TIME_FEATURE], format='%d.%m.%Y %H:%M', errors='ignore')
    df = df.set_index(TIME_FEATURE)
    df = df.astype(dtype)
    df = df.dropna(axis=0, how='all').dropna(axis=1, how='all')
    
    # dump to dump
    data_pickle[pars] = df
    write(file_name_pickle, data_pickle)
    return df


def load_all(load_test=False):
    return [load(datafile_name) for datafile_name in (DATAFILE_NAMES if load_test else DATAFILE_NAMES[:-1])]


def train_split(df, test_size=0.5):
    n = int(df.shape[0] * (1 - test_size))
    return df.iloc[:n], df.iloc[n:]


def train_split_stratified(df, test_size=0.5):
    return train_test_split(df, test_size=test_size, random_state=RANDOM_SEED, stratify=df[TARGET_FEATURE])


def target_split(df):
    if isinstance(df, pd.DataFrame) and TARGET_FEATURE in df.columns:
        return df.drop(columns=TARGET_FEATURE), df[TARGET_FEATURE]
    else:
        return df, None


def is_dundler(name):
    return len(name) > 4 and name.isascii() and name.startswith('__') and name.endswith('__')


def module_to_pickle(module):
    res = []
    for name in dir(module):
        obj = getattr(module, name)
        if not any([is_dundler(name), isinstance(obj, module.__class__)]):
            res.append(obj)
    write(r'dump/{}.pickle'.format(module.__name__), res)


def all_modules_in_folder_to_pickle(folder):
    from importlib.util import spec_from_file_location, module_from_spec

    def get_module_from_path(path):
        path = Path(path)
        return module_from_spec(spec_from_file_location(path.stem, path))

    files = [x for x in Path(folder).glob('*.py') if not is_dundler(str(x).rstrip('.py'))]
    modules = [get_module_from_path(x) for x in files]
    for module in modules:
        module_to_pickle(module)


def mode_sequencer(df):
    df = df.dropna()
    x, y = target_split(df)
    yd = y.diff().reset_index(drop=True)
    ind = yd[yd != 0].index
    for start, end in zip(ind, ind[1:]):
        yield df.iloc[start:end]
    
        
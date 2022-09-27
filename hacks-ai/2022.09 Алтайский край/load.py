from math import nan
import itertools
from pathlib import Path
from copy import deepcopy
import pickle

import pandas as pd
from sklearn.model_selection import train_test_split

from parameters import (PATH_TO_DATA, TEST_DATA_FILE, TRAIN_DATA_FILE, TEST_SIZE,
                        RANDOM_SEED, COLS_RENAMING_DICT, TARGET_FEATURE, TARGET_REPLACER, INVERSE_TARGET_REPLACER)
from processing import start_processing
from preparing import (MyOheHotEncoder, MyOrdinalEncoder, MyMinMaxScaler, ColumnsSorter,
                       EmptyColFiller, MyPolynomialFeatures, ordinal_encoding, one_hot_encoding)



def read_data(all_=False, test=False, split_to_f_t=False, split_to_train_test=False,
              retain_target=True, class_=None, processing=False):
    res = {'f': [], 't': []}
    if all_:
        res['f'] = pd.concat([pd.read_csv(Path(PATH_TO_DATA) / Path(TRAIN_DATA_FILE)),
                              pd.read_csv(Path(PATH_TO_DATA) / Path(TEST_DATA_FILE))])
    else:
        res['f'] = pd.read_csv(Path(PATH_TO_DATA) / Path(TEST_DATA_FILE if test else TRAIN_DATA_FILE))
    res['f'] = res['f'].rename(columns=COLS_RENAMING_DICT)
    res['f'] = res['f'].set_index('id')
    res['f'][TARGET_FEATURE] = res['f'][TARGET_FEATURE].replace(TARGET_REPLACER)
    if processing:
        res['f'] = start_processing(res['f'])
    if class_ is not None and TARGET_FEATURE in res['f'].columns:
        if pd.isna(class_):
            res['f'] = res['f'][res['f'][TARGET_FEATURE].isna()]
        else:
            res['f'][TARGET_FEATURE] = (res['f'][TARGET_FEATURE] == class_).astype(int)
    res['f'] = [res['f']]
    if not test:
        if split_to_train_test and not pd.isna(class_):
            res['f'] = train_test_split(res['f'][0], test_size=TEST_SIZE, random_state=RANDOM_SEED, stratify=res['f'][0][TARGET_FEATURE])
        if split_to_f_t:
            res['t'] = [f[TARGET_FEATURE] for f in res['f']]
        if not retain_target:
            res['f'] = [f.drop(columns=[TARGET_FEATURE]) for f in res['f']]
    if not split_to_f_t:
        return res['f'] if len(res['f']) > 1 else res['f'][0]
    else:
        res['t'] = [t.astype(int) for t in res['t']]
        return list(itertools.zip_longest(*(res.values())))


def write(file_name, data):
    with open(file_name, 'wb') as handle:
        pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)


def read(file_name):
    with open(file_name, 'rb') as handle:
        return pickle.load(handle)


def split_to_f_t(df):
    fs = df[~df[TARGET_FEATURE].isna()]
    f = []
    for i in range(len(TEST_SIZE)):
        test_size = TEST_SIZE[i] / (1 - sum(TEST_SIZE[:i]))
        fs, f2 = train_test_split(fs, test_size=test_size, random_state=RANDOM_SEED, stratify=fs[TARGET_FEATURE])
        f.append(f2)
    f.append(fs)
    return [(x.drop(columns=TARGET_FEATURE), x[TARGET_FEATURE]) for x in reversed(f)]


def get_full_prepared_data_with_upsample(n=[None, None, None]):
    return_data_file = Path('dump/temp_data.pickle')
    if (return_data_file.exists()
        and (return_data_file.stat().st_mtime > Path('processing.py').stat().st_mtime
             and return_data_file.stat().st_mtime > Path('parameters.py').stat().st_mtime
             and return_data_file.stat().st_mtime > Path('load.py').stat().st_mtime)):
        return read(return_data_file)
        
    
    f0 = read_data(all_=True, processing=True, split_to_f_t=False, split_to_train_test=False, retain_target=True, class_=None)
    f, e = ordinal_encoding(f0)
    cols_to_drop = []
    sub_cols_to_drop = []
    
    # cols_to_drop = ['group_code', 'years_old', 'language', 'school', 'school_location', 'school_finish_year',
    #                 'region', 'city', 'community', 'has_mother', 'has_father', 'relativies_country',
    #                 'countryside', 'foreign', 'start_year_val', 'birthday_year', 'birthday_month', 'years_old',
    #                 'mean_mark_type1', 'id'] + [x for x in f.columns if 'group_code' in x]
    # cols_to_drop = ['language', 'school', 'school_location', 'school_finish_year',
    #                 'region', 'city', 'community', 'relativies_country',
    #                 'foreign', 'start_year_val', 'birthday_month', 'years_old',
    #                 'mean_mark_type1'] + [x for x in f.columns if 'group_code' in x or 'diff_between_school_n_start' in x]
    # cols_to_drop = ['id'] + [x for x in f.columns if 'group_code' in x]
    if cols_to_drop:
        f = f.drop(columns=[x for x in cols_to_drop if x in f.columns])
    
    sub_cols_to_drop = ('scale_fun', )
    f = f.drop(columns=[x for x in f.columns if any(y in x for y in sub_cols_to_drop)])
    
    ft = split_to_f_t(f[~f[TARGET_FEATURE].isna()])
    
    if not all([x is None for x in n]):
        raise NotImplementedError
        f0, t0, f2, t2 = ft
        f00 = pd.concat([f0, t0], axis=1)
        f00_0, f00_1, f00_2 = f00[t0 == 0], f00[t0 == 1], f00[t0 == 2]
        f00 = pd.concat([f00_0.sample(n[0] if n[0] else f00_0.shape[0], random_state=RANDOM_SEED),
                         f00_1.sample(n[1] if n[1] else f00_1.shape[0], random_state=RANDOM_SEED),
                         f00_2.sample(n[2] if n[2] else f00_2.shape[0], replace=True, random_state=RANDOM_SEED)])
        f0 = f00.drop(columns=[TARGET_FEATURE])
        t0 = f00[TARGET_FEATURE]
        ft = (f0, t0, f2, t2)

    r = f, ft, f[f[TARGET_FEATURE].isna()].drop(columns=[TARGET_FEATURE])
    write(return_data_file, r)
    return r
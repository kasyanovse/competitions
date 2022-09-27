from math import nan
from functools import partial
from difflib import SequenceMatcher
from multiprocessing import Pool

import numpy as np
import numpy.random
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from imblearn.over_sampling import RandomOverSampler

from parameters import THREADS, TARGET_FEATURE, TARGET_REPLACER, INVERSE_TARGET_REPLACER, RANDOM_SEED
import processing_data
from preparing import ordinal_encoding


def fillna_by_most_popular(series):
    return series.fillna(series.value_counts().sort_values().index[-1])


def fillna_by_zero_in_category(series):
    if series.cat.categories.dtype in ['int64', 'float64']:
        return (series.cat.codes + 1).astype('category')
    raise TypeError(f"categories must be int64 or float64 but it is {series.cat.categories.dtype}")


def replace_by_substring(series, replace_dict):
    series = series.copy()
    s = series.dropna()
    for k, v in replace_dict.items():
        s = s.str.replace(k, v)
    series.loc[s.index] = s
    return series


def category_by_substring(series, cat_dict):
    s = series.dropna()
    for k, v in cat_dict.items():
        s0 = s.str.contains(k)
        series.loc[s0[s0].index] = v
    return series.astype('category')


def delete_substr_from_str(s, substrs):
    for substr in substrs:
        s = s.replace(substr, '')
    return s


def retain_alnum(s, ignore_non_string=True):
    if isinstance(s, str):
        s = ''.join([x if x.isalnum() else ' ' for x in s])
        return ' '.join(s.split())
    else:
        return s


def similar(str1, str2):
    str1 = str1[:max(len(str1), len(str2))]
    str2 = str2[:max(len(str1), len(str2))]
    return SequenceMatcher(None, str1, str2).ratio()


def compare_strings(str1, str2):
    str1, str2 = retain_alnum(str1).lower(), retain_alnum(str2).lower()
    res = dict()
    for s1 in str1.split():
        res[s1] = 1e9
        for s2 in str2.split():
            new_res = similar(s1, s2)
            if new_res < res[s1]:
                res[s1] = new_res
    return sum(res.values())


def compare_strings_unordered(str1, str2):
    return max(compare_strings(str1, str2), compare_strings(str2, str1))


def find_similar_strings(strs, edge=1):
    res = []
    for ind1 in range(len(strs) - 1):
        for ind2 in range(ind1 + 1, len(strs)):
            if isinstance(strs[ind1], str) and isinstance(strs[ind2], str):
                res.append((strs[ind1], strs[ind2], compare_strings_unordered(strs[ind1], strs[ind2])))
    res = pd.DataFrame(res, columns=[1, 2, 'val'])
    return res[res['val'] < edge].sort_values('val')

def find_similar_strings_par(strs, edge):
    fun = partial(find_similar_strings, edge=edge)
    with Pool(THREADS) as p:
        res = pd.concat(p.imap(fun, strs, (len(strs) // THREADS) + 1))
    return res.sort_values('val')


def replace_by_sub_str(f, substr, new_str):
    if new_str is None:
        new_str = substr

    def rep(x):
        if not pd.isna(x):
            if ' ' in substr and substr in x:
                return new_str
            if any([y == substr for y in x.split()]):
                return new_str
        return x
    return f.apply(rep)
    
    
def replace_by_another_non_nan_col(df, col1, col2, replace_dict=None):
    selector = df[col1].isna() & (~df[col2].isna())
    selector = selector[selector].index
    df.loc[selector, col1] = df.loc[selector, col2].str.lower().replace(replace_dict) if replace_dict else df.loc[selector, col2]
    return df


def replace_by_word(f, replace_vals, drop_edge=10):
        for val in replace_vals:
            f = replace_by_sub_str(f, *val)
        temp = f.value_counts()
        f[f.isin(temp[temp < drop_edge].index)] = nan
        f = f.replace('', nan)
        f = f.replace('nan', nan)
        return f


def merge_small_cat(df, col, num_of_instance, val_for_replace):
    temp = df[col].value_counts() < num_of_instance
    temp = temp[temp].index
    df.loc[df[col].isin(temp), col] = val_for_replace
    return df
    

def knn_feature(neighbours, df, target_feature, n_jobs=-1, **kwargs):
    df = df.copy()
    if target_feature != TARGET_FEATURE:
        target = df[TARGET_FEATURE]
        df = df.drop(columns=[TARGET_FEATURE])
    df, _ = ordinal_encoding(df)
    df1 = df[~df[target_feature].isna()]
    
    if False:
        os = RandomOverSampler(sampling_strategy='not majority', random_state=RANDOM_SEED)
        x, y = os.fit_resample(df1.drop(columns=[target_feature]), df1[target_feature])
    else:
        x, y = df1.drop(columns=[target_feature]), df1[target_feature]
    
    if True:
        x = pd.concat([x, y], axis=1)
        x = x.sample(frac=0.5)
        x, y = x.drop(columns=[target_feature]), x[target_feature]
    
    model = KNeighborsClassifier(n_neighbors=neighbours, n_jobs=n_jobs, weights='distance')
    model.fit(x, y)
    return pd.Series(model.predict(df.drop(columns=[target_feature])), index=df.index).astype(int).astype('category')

def family_of_knn_features(df, name, target_feature, n_counts=(5, 20, 50)):
    if target_feature in df.columns and any(~df[target_feature].isna()):
        temp = dict()
        for i in n_counts:
            temp[f'{name}_{i}'] = knn_feature(i, df, target_feature=target_feature)
        for k, v in temp.items():
            df[k] = v
    return df


def start_processing(df, *args):
    df = df.copy()
    
    df['gender'] = category_by_substring(df['gender'].apply(lambda x: x if pd.isna(x) else x[0].lower()), {'ж': 1, 'м': 0})
    df['gender'] = fillna_by_most_popular(df['gender']).astype(int).astype('category').cat.set_categories([0, 1])

    df['condition'] = df['condition'].replace('ЛН', 'ДН')
    df['condition'] = df['condition'].astype('category')
    
    
    def get_digits_from_number(x, start_digit, count_of_digit, max_number=100000):
        f1 = max_number / (10 ** start_digit)
        f2 = max_number / (10 ** (start_digit + count_of_digit))
        return (x - (x // f1) * f1 - (x % f2)) // f2
    
    # df['group_code'] = df['group_code'].astype(int)
    # for start_digit in range(0, 4):
    #     for digits_count in range(1, 3):
    #         if start_digit + digits_count <= 5:
    #             # if start_digit not in (0, 1, 2) or digits_count > 2:
    #             #     continue
    #             name = f'group_code_add_{start_digit}_{digits_count}'
    #             df[name] = get_digits_from_number(df['group_code'], start_digit, digits_count).astype(int)
    #             if len(df[name].unique()) < 25:
    #                 df[name] = df[name].astype('category')
    
    for i in [1, 2, 3, 4]:
        df[f'group_code_add_{i}'] = (df['group_code'] // (10 ** i) % 10).astype('category')
    
    df['group_code_num'] = ((df['group_code'] - df['group_code'].min()) / (df['group_code'].max() - df['group_code'].min()) * 20).astype(float)
    df['group_code'] = df['group_code'].astype(int).astype('category')
    
    # for i in [3, 4, 5]:
    #     df[f'id_add_{i}'] = (df.index // (10 ** i) % 10).astype('category')
    # df[f'id_add'] = (df.index // 100 % 1000).astype('category')
    # df['id'] = df.index.astype(int)

    df['language'] = category_by_substring(df['language'].str.lower(), {'анг': 0, 'нем': 1, 'фр': 2, 'рус': 3})
    df['language'] = df['language'].replace({3: 2}).fillna(2).astype(int).astype('category')

    df['start_year_val'] = (df['start_year'] % 100)
    df['start_year_val'] -= df['start_year_val'].min()
    df['start_year'] = (df['start_year'] % 100) + 2000
    df['birthday'] = pd.to_datetime(df['birthday'], format='%Y-%m-%d')
    df['birthday_year'] = df['birthday'].dt.year
    df['birthday_year'] = df['birthday_year'].where(df['birthday_year'] >= 1980, 1980)
    df['birthday_year'] = df['birthday_year'].where(df['birthday_year'] <= 2002, 2002)
    df['birthday_year'] = df['birthday_year'].astype(int).astype('category')
    df['birthday_month'] = df['birthday'].dt.month.astype(int).astype('category').cat.set_categories(list(range(1, 13)))
    
    df['years_old'] = df['start_year'] - df['birthday'].dt.year
    df['years_old'] = df['years_old'].where(df['years_old'] > 16, 16)
    # df['years_old'] = df['years_old'].where(df['years_old'] < 35, 35)
    df['years_old'] = df['years_old'].astype(int)
    
    df['diff_between_school_n_start'] = df['start_year'] - df['school_finish_year']
    # df['diff_between_school_n_start'] = df['diff_between_school_n_start'].where(df['diff_between_school_n_start'] < 30, 30)
    df['diff_between_school_n_start'] = df['diff_between_school_n_start'].where(df['diff_between_school_n_start'] >= 0, 0)
    # df['diff_between_school_n_start'] = df['diff_between_school_n_start'].astype(int)
    
    df['start_year'] = df['start_year'].where(df['start_year'] >= 2012, 2012)
    df['start_year'] = df['start_year'].where(df['start_year'] <= 2019, 2019)
    df['start_year'] = df['start_year'].astype(int).astype('category')
    df['start_year'] = df['start_year'].cat.set_categories([2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019])
    # df['start_year'] = df['start_year'].cat.set_categories([2001, 2003, 2004, 2005, 2006, 2007, 2008, 2009, 2010, 2011,
    #                                                         2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020])
    
    df['mean_mark_type1'] = (df['mean_mark'] > 10).fillna(False).astype(int).astype('category')
    df['mean_mark'] = df['mean_mark'].where(df['mean_mark'] > 5,
                                            (df['mean_mark'] * 15).where(df['mean_mark'] < 4, (df['mean_mark'] - 4) * 40 + 60))
    for i in range(20, 0, -1):
        df['mean_mark'] = df['mean_mark'].where(df['mean_mark'] < i * 100 + 1, df['mean_mark'] / i)
    df['mean_mark'] = df['mean_mark'].where(df['mean_mark'] < 101, df['mean_mark'].median())
    df['mean_mark'] = df['mean_mark'].where(df['mean_mark'] > 30, 30)
    df['mean_mark'] = df['mean_mark'].astype(int)
    
    df['mean_mark_type2'] = pd.concat([df['mean_mark'] > df['mean_mark'].quantile(q) for q in (0.5, 0.75, 0.85, 0.9)], axis=1).sum(axis=1).astype(int).astype('category')
    df['mean_mark_add1'] = ((df['mean_mark'] - df['mean_mark'].median()) / 500) ** 2
    df['mean_mark_add2'] = (((df['mean_mark'] - df['mean_mark'].median()) / 500).abs() - ((df['mean_mark'] - df['mean_mark'].median()) / 500).max()) ** 2
    
    scaling_funs = {'r': lambda x: (x - x.max()).abs(),
                    'sin': lambda x: np.sin(x),
                    '2': lambda x: x ** 2,
                    'exp3': lambda x: np.exp(-x / 3),
                    'exp5': lambda x: np.exp(-x / 5),
                    'expexp': lambda x: np.exp(-x / 2) + np.exp(-x / 5),
                    'rexp3': lambda x: np.exp(-(x - x.max()).abs() / 3),
                    'rexp5': lambda x: np.exp(-(x - x.max()).abs() / 5),
                    'sqrt': lambda x: np.sqrt(x.abs()),
                    'log': lambda x: (x - x.min()) / (x.max() - x.min()) * 99 + 1}
    for col in ('years_old', 'diff_between_school_n_start', 'start_year_val', 'mean_mark', 'group_code_num'):
        if col in df.columns:
            for fun_name, fun in scaling_funs.items():
                df[f"{col}_{fun_name}_scale_fun"] = fun(df[col])
    # scaling_funs = {'r': lambda x: (x - x.max()).abs(),
    #                 '2': lambda x: x ** 2,
    #                 'exp3': lambda x: np.exp(-x / 3),
    #                 'exp5': lambda x: np.exp(-x / 5),
    #                 'expexp': lambda x: np.exp(-x / 2) + np.exp(-x / 5),
    #                 'rexp3': lambda x: np.exp(-(x - x.max()).abs() / 3),
    #                 'rexp5': lambda x: np.exp(-(x - x.max()).abs() / 5),
    #                 'sqrt': lambda x: np.sqrt(x.abs())}
    # for col in ('diff_between_school_n_start', ):
    #     if col in df.columns:
    #         for fun_name, fun in scaling_funs.items():
    #             df[f"{col}_{fun_name}_scale_fun"] = fun(df[col])

    
    # countries and regions
    df['country'] = category_by_substring(df['country'].str.lower(), processing_data.countries).replace(processing_data.countries_large).astype('object')
    df['relativies_country'] = category_by_substring(df['relativies_country'].str.lower(), processing_data.countries).astype('object').replace(processing_data.countries_large)
    df['region'] = df['region'].str.lower().replace(processing_data.regions).replace(processing_data.regions_large)
    df = replace_by_another_non_nan_col(df, 'country', 'region', processing_data.regions_to_country)
    df = replace_by_another_non_nan_col(df, 'country', 'city', processing_data.city_small_list_to_country)
    df = replace_by_another_non_nan_col(df, 'region', 'country', processing_data.country_to_regions)
    df = replace_by_another_non_nan_col(df, 'relativies_country', 'country')
    df = replace_by_another_non_nan_col(df, 'country', 'relativies_country')
    df['country'] = df['country'].fillna('россия')
    
    df['country'] = df['country'].astype('category')
    df['relativies_country'] = df['relativies_country'].astype('category')
    df['region'] = df['region'].astype('category')

    # countryside
    city_to_countryside = {k.lower(): v for k, v in processing_data.city_to_countryside.items()}
    df = replace_by_another_non_nan_col(df, 'countryside', 'city', city_to_countryside)
    df['countryside'] = df['countryside'].fillna(0)
    df['countryside'] = df['countryside'].astype(int).astype('category').cat.set_categories([0, 1])
    
    # foreign
    df = replace_by_another_non_nan_col(df, 'foreign', 'country', {'казахстан': 1, 'россия': 0, 'ср азия': 1, 'китай': 1})
    df['foreign'] = df['foreign'].fillna(0)
    df['foreign'] = df['foreign'].astype(int).astype('category').cat.set_categories([0, 1])
    
    # city
    df.loc[df['country'] == 'китай', 'city'] = 'китай'
    df['city'] = replace_by_word(df['city'].str.lower().apply(retain_alnum), processing_data.city_replace_vals, 50)
    df.loc[df['country'] == 'китай', 'school_location'] = 'китай'
    df['school_location'] = replace_by_word(df['school_location'].str.lower().apply(retain_alnum), processing_data.city_replace_vals, 50)
    
    df['city'] = df['city'].astype('category')
    df['school_location'] = df['school_location'].astype('category')
    
    # school
    temp = df['school'].str.lower().apply(retain_alnum)
    df['school_type'] = nan
    for k, v in processing_data.school_filter.items():
        temp1 = pd.Series(False, index=df.index)
        for iv in v:
            temp1 = temp1 | temp.str.contains(iv)
        df.loc[temp1, 'school_type'] = k
    df = merge_small_cat(df, 'school_type', 150, 'other')
    df['school_type'] = df['school_type'].fillna('other')
    df['school_type'] = df['school_type'].astype('category')
    
    
    def prepare_school_in_city(df, city, replace_vals, edge):
        selector = pd.Series(True, index=df.index) if city is None else df['school_location'] == city
        if replace_vals is not None:
            replace_vals = sorted(replace_vals, key=lambda x: len(x[0]), reverse=True)
            temp = replace_by_word(df[selector]['school'].str.lower().apply(retain_alnum), replace_vals, edge)
            temp = temp.dropna()
            df.loc[temp.index, 'school'] = temp
        temp = replace_by_word(df[selector]['school'].str.lower().apply(retain_alnum),
                               processing_data.school_common_replace_vals + processing_data.school_ordinal_replace_vals, 10)
        df.loc[temp.index, 'school'] = temp
        return df
    
    df = prepare_school_in_city(df, 'барнаул', processing_data.school_barn_replace_vals, 50)
    df = prepare_school_in_city(df, 'бийск', processing_data.school_biysk_replace_vals, 50)
    df = prepare_school_in_city(df, 'новоалтайск', processing_data.school_novoaltaysk_replace_vals, 50)
    df = prepare_school_in_city(df, None, None, 50)
    
    df['school'] = df['school'].astype('category')

    # community
    df['community'] = df['community'].where(~df['community'].isna() | (df['community'].isna() & df['city'].isna()), df['city'] != 'барнаул')
    df['community'] = df['community'].fillna(0)
    df['community'] = df['community'].astype(int).astype('category').cat.set_categories([0, 1])
    
    # faculty
    df = merge_small_cat(df, 'faculty', 150, 99)
    df['faculty'] = df['faculty'].astype(int).astype('category')

    # other
    df['guardianship'] = df['guardianship'].astype('category').cat.set_categories([0, 1])
    df['has_father'] = df['has_father'].astype('category').cat.set_categories([0, 1])
    df['has_mother'] = df['has_mother'].astype('category').cat.set_categories([0, 1])
    df['has_full_family'] = ((df['has_father'] == 1) & (df['has_mother'] == 1)).astype(int).astype('category').cat.set_categories([0, 1])
    df['has_not_family'] = ((df['has_father'] == 0) & (df['has_mother'] == 0)).astype(int).astype('category').cat.set_categories([0, 1])
    
    for col in df.select_dtypes(include='category').columns:
        df[col] = df[col].astype('object')
        if df[col].isna().any():
            df[col] = df[col].fillna('nan')
        df[col] = df[col].astype('category')
    
    df = df.drop(columns=['school_finish_year', 'pension', 'guardianship', 'birthday', 'birthday_month', 'years_old'])
    
    df = family_of_knn_features(df, 'k', TARGET_FEATURE)
    # df = family_of_knn_features(df, 'sub_group_code', 'group_code')
    
    def temp(x):
        x = x.value_counts()
        if len(x) == 0:
            return nan
        return x.index[0]

    df['same_group_target'] = df['group_code'].replace(df.groupby('group_code')[TARGET_FEATURE].agg(temp).fillna(0).to_dict()).astype(int).astype('category')
    return df
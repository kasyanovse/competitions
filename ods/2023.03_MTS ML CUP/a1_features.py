""" function for generating features for 1 approach """

from collections import Counter

import numpy as np
import pandas as pd
from tqdm import tqdm

from settings import path
from load import load, write, load_target, load_col

from process_data import (dataframe_group_to_dict_by_user, simple_process_category,
                          simple_process, process_url_n_set_to_data,
                          get_new_features_from_date, get_new_features_from_part_of_day,
                          process_manufacturer, dataframe_group_indexes_to_dict_by_user,
                          get_additional_data_about_url, DFD, process_url_n_set_to_data_full)


DATA_PATH = path['data']['approach_1']
COMMON_FEATURE_PATH = DATA_PATH / 'common_features.pickle'
COMMON_FEATURES_DICT_PATH = DATA_PATH / 'common_features_dict.pickle'
FUNS = dict()
CACHED_COLS = dict()


## common data for features


def get_common_feature():
    df = load(COMMON_FEATURE_PATH)
    if df is None:
        features= [simple_process(load_col('user_id')),
                   # process_url_n_set_to_data(load_col('url_host')),
                   # get_new_features_from_date(load_col('date')),
                   # get_new_features_from_part_of_day(load_col('part_of_day')),
                   # process_manufacturer(load_col('cpe_manufacturer_name')),
                   # simple_process_category(load_col('region_name')),
                   # simple_process_category(load_col('city_name')),
                   # simple_process_category(load_col('cpe_model_name')),
                   # simple_process_category(load_col('url_host')),
                   # simple_process(load_col('price')),
                   # simple_process(load_col('date')),
                   # simple_process(load_col('request_cnt')),
                   # simple_process_category(load_col('cpe_type_cd')),
                   ]
        df = pd.concat([features.pop() for _ in range(len(features))], axis=1)
        df = df.sort_values('user_id').reset_index(drop=True)
        write(COMMON_FEATURE_PATH, df)
    return df

def get_common_feature_dict(force_dict_preparing=False):
    return dataframe_group_indexes_to_dict_by_user(COMMON_FEATURES_DICT_PATH, get_common_feature)


## data for features (cols like in common data for features)


def cache_data_for_feature(fun, data_path=DATA_PATH, cached_cols=CACHED_COLS):
    file_name = data_path / f"d_{fun.__name__}.pickle"
    def temp(fun=fun, file_name=file_name, cached_cols=cached_cols):
        df = load(file_name)
        if df is None:
            user_id = cached_cols['user_id'] if 'user_id' in cached_cols else simple_process(load_col('user_id'))
            df = pd.concat([user_id, fun()], axis=1)
            df = df.sort_values('user_id').reset_index(drop=True)
            df = df.drop(columns='user_id')
            write(file_name, df)
        return df
    return temp


def check_df_is_appropriate_for_dfis_and_insert_it(dfis, df):
    if (type(dfis.df.index) != type(df.index)
        or dfis.df.index.start != df.index.start
        or dfis.df.index.stop != df.index.stop
        or dfis.df.index.step != df.index.step):
        raise ValueError('that dfis is not appropriate for df')
    dfis.df = df
    return dfis


@cache_data_for_feature
def data_from_url_simple(cached_cols=CACHED_COLS):
    col = 'simple_url_processed'
    return cached_cols[col] if col in cached_cols else process_url_n_set_to_data(load_col('url_host'))


@cache_data_for_feature
def data_from_url(cached_cols=CACHED_COLS):
    return process_url_n_set_to_data_full(load_col('url_host'))


@cache_data_for_feature
def data_from_date(cached_cols=CACHED_COLS):
    return get_new_features_from_date(load_col('date'))


## feature processing


def prepare_feature(fun, funs=FUNS, data_path=DATA_PATH):
    file_name = data_path / f"f_{fun.__name__}.pickle"
    def temp(dfis, file_name=file_name, fun=fun, allow_additional_calculations=True):
        def calc_for_user_id(dfis, file_name=file_name):
            print(file_name)
            result = [fun(dfis[key]) for key in tqdm(dfis.keys())]

            add_kwargs = dict()
            add_kwargs |= {'index': dfis.keys()}
            for type_ in ('category', 'float32', 'int32'):
                if file_name.stem.endswith(type_):
                    add_kwargs |= {'dtype': type_}
                    break
            if isinstance(result[0], dict):
                result = pd.DataFrame(result, **add_kwargs)
            else:
                add_kwargs |= {'name': file_name.stem}
                result = pd.Series(result, **add_kwargs)
            return result
        
        new_dfis = fun(dfis)
        if new_dfis is not None:
            dfis = new_dfis
        
        if file_name.exists():
            result = load(file_name)
            delta_user_ids = [x for x in dfis.keys() if x not in result.index]
            if delta_user_ids and allow_additional_calculations:
                new_res = calc_for_user_id({user_id: dfis[user_id] for user_id in delta_user_ids})
                result = pd.concat([result, new_res]).astype(new_res.dtypes)
                write(file_name, result)
            elif not allow_additional_calculations:
                user_ids = list(result.index)
        else:
            result = calc_for_user_id(dfis)
            write(file_name, result)
        return result.loc[list(dfis.keys())]

    funs[file_name.stem] = temp
    return temp


## features


# @prepare_feature
# def domain_category(dfi):
#     if isinstance(dfi, DFD):
#         return check_df_is_appropriate_for_dfis_and_insert_it(dfi, data_from_url_simple())
#     res = dict()
#     for i in (1, 2):
#         temp = dfi[i].value_counts()
#         for j in (0, 1, -5, -4, -3, -2, -1):
#             res[f"domain_{i}_{'m' if j < 0 else ''}{abs(j) + 1}_category"] = temp.index[j]
#     return res

# @prepare_feature
# def count_of_domains_in_quantile_int32(dfi):
#     if isinstance(dfi, DFD):
#         return check_df_is_appropriate_for_dfis_and_insert_it(dfi, data_from_url_simple())
#     res = dict()
#     temp = (dfi[2].value_counts() / dfi.shape[0]).cumsum()
#     for quantile in (0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9):
#         res[f"count_of_domains_in_quantile_{int(quantile*100)}"] = (temp < quantile).sum()
#     return res

for col in ('turbopages', ):
    def temp(dfi, col=col):
        if isinstance(dfi, DFD):
            return check_df_is_appropriate_for_dfis_and_insert_it(dfi, data_from_url_simple())
        return dfi[col].astype(int).mean()
    temp.__name__ = f"share_of_{col}_sites_float32"
    prepare_feature(temp)

# @prepare_feature
# def dow_analisys_float32(dfi):
#     if isinstance(dfi, DFD):
#         return check_df_is_appropriate_for_dfis_and_insert_it(dfi, data_from_date())
#     temp = dfi.value_counts()
#     res = {'most_loaded_dow': temp.iloc[0],
#            'less_loaded_dow': temp.iloc[-1],
#            'mean_requests_in_wd': temp.loc[[0, 1, 2, 3, 4]].mean(),
#            'mean_requests_in_we': temp.loc[[5, 6]].mean(),
#           }
#     return res

# @prepare_feature
# def most_loaded_part_of_dow_category(dfi):
#     if isinstance(dfi, DFD):
#         return check_df_is_appropriate_for_dfis_and_insert_it(dfi,
#                       pd.concat([data_from_date(),
#                                  get_new_features_from_part_of_day(load_col('part_of_day')),
#                                  ], axis=1))
#     res = dict()
#     for dow in range(7):
#         temp = dfi[dfi['dow'] == dow]['part_of_day'].value_counts()
#         for n in (0, 1):
#             res[f"most_loaded_part_of_dow_{n}_{dow}"] = temp.index[n]
#     return res


# that list is defined in 01_first_look.ipynb
sites_with_different_audience = ['samsungapps', 'yastatic', 'youtube', 'ftd', 'twitter', 'userapi', 'ytimg', 'tivizor', 'steamcommunity', 'sports', 'vtb', 'vazhno', 'auto', 'job', 'matchtv', 'tinkoff', 'smotret-video', 'moevideo', 'sberbank', 'duckduckgo', 'rocketme', 'loveome', 'alfabank', 'i-trailer', 'iz', 'directadvert', 'realsrv', 'eagleplatform', '1tv', 'betweendigital', 'streamalloha', 'mos', 'aliexpress', 'instagram', 't', 'smotrim', 'buzzoola', 'smi2', 'gazeta', 'film', 'dni', 'adfox', 'gnezdo', 'pozdravok', 'wargaming', 'safebrowsdv', 'mk', 'advmusic', 'rt', 'pinimg', 'infotime', 'googleapis', 'povar', 'icloud', 'inplayer', 'gosuslugi', 'viqeo', 'rg', 'ampproject', 'ok', 'mail', 'news-fancy', '7days', 'bit', 'playreplay', 'mediametrics', 'relap', 'whatsapp', 'adwile', 'riafan', 'kost', 'ria', 'rambler', 'apteka', 'apple', 'prodoctorov', 'viadata', 'glavnoe', 'howto-news', 'ren', 'vgtrk', '4251', 'profile', 'avito', 'sunlight', 'eda', 'sbrf', 'radiokp', 'blitz', 'appnext', 'ura', 'vk', 'sport-express', 'doubleclick', 'aviasales', 'zdorovcom', 'discord', 'trafficfactory', 'irecommend', 'googlesyndication', 'teleprogramma', 'yandex', 'mts', 'kpcdn', 'tass', 'skwstat', 'razdvabm', 'synchroncode', '24smi', 'ya', 'wi-fi', 'amazonaws', 'drive2', 'news-sphere', 'mycdn', 'stiven-king', 'thesame', 'zoon', 'wikipedia', 'showjet', 'championat', 'sport24', 'udipedia-new', 'ixbt', 'lenta', '1000', '2mdn', 'i24-7-news', 'rtb', 'hhcdn', 'lookmeet', 'infox', 'drom', 'sportbox', 'google', 'videoroll']
sites_with_different_audience += ['yandex', 'userapi', 'doubleclick', 'mail', 'vk', 'ytimg', 'yastatic', 'sberbank', 'apple', 'google', 'adfox', 'instagram', 'googlesyndication', 'adriver', 'ok', 'icloud', 'buzzoola', '2mdn', 'betweendigital', 'googleapis', 'relap', 'otm-r', 'tinkoff', 'rambler', 'gosuslugi', 'amazonaws', 'rtb', 'mts', 'film', 'avito', 'playreplay', 't', 'facebook', 'weborama', 'sape', 'mycdn', 'smi2', 'pinimg', 'lenta', 'bidvol', 'moevideo', 'sbrf', 'aliexpress', 'skwstat', 'vtb', 'duckduckgo', 'job', 'thesame', 'samsungapps', 'realsrv']
sites_with_different_audience += ['kinostream', 'blitz', 'kinomans', 'buzzoola', 'mail', 'viqeo', 'howto-news', 'rambler', 'adriver', 'mycdn', 'ria', 'kinoaction', 'sportbox', 'smotrim', 'pinimg', 'teleprogramma', 'thesame', 'ixbt', 'infotime', 'genius', 'leroymerlin', 'ya', 'ftd', 'safebrowsdv', 'auto', 'xn--90adear', 'lenta', 'postupi', 'zoon', 'news-fancy', 'instagram', 'profile', '24smi', 'zdorovcom', 'drive2', 'rtb', 'tass', 'adfox', 'mts', 'doubleclick', 'eagleplatform', 'appnext', 'discord', 'newzfeed', 'kost', 'sport-express', 'bit', '4251', 'sbrf', 'videoroll', 'betweendigital', 'wikipedia', 'kpcdn', 'ytimg', 'tinkoff', 'matchtv', 'otm-r', 'mi7', 't', 'gosuslugi', 'youtube', 'samsungapps', 'vazhno', 'duckduckgo', 'relap', 'viiadr', 'lookmeet', 'ura', 'udipedia-new', 'amazonaws', 'steampowered', 'mk', 'vz', 'rt', 'sports', 'aliexpress', 'tivizor', '1tv', 'mediametrics', 'mos', 'aviasales', 'prodoctorov', 'cian', 'text-pesni', 'apteka', 'yastatic', 'gazeta', 'hh', 'loveome', 'moevideo', 'i-trailer', 'ampproject', 'povar', 'gnezdo', 'discordapp', 'googlesyndication', 'infox', 'whatsapp', 'glavnoe', 'ok', 'vk', 'drom', 'sport24', '2mdn', 'twitter', 'weborama', 'googleapis', 'google', 'smi2', 'trafficfactory', 'sunlight', '1000', 'adwile', 'playreplay', 'apple', 'news-sphere', 'championat', 'radiokp', 'userapi', 'eda', 'rg', 'irecommend', 'iz', 'streamalloha', 'hhcdn', 'wargaming', 'steamcommunity', 'i24-7-news', 'viadata', 'vgtrk', 'ren', 'showjet', 'riafan', 'pochtabank', 'filmskino', 'pluso', 'realsrv', 'yandex', 'sberbank', 'wi-fi', 'sdamgia', 'advmusic', 'qiwi', 'otzovik', 'razdvabm', 'icloud', 'avito', 'kinopoisk', 'pozdravok', 'directadvert', 'stiven-king', 'skwstat', 'alfabank', 'inplayer', 'vtb']
sites_with_different_audience += ['yandex', 'userapi', 'doubleclick', 'mail', 'vk', 'ytimg', 'yastatic', 'sberbank', 'apple', 'google', 'adfox', 'instagram', 'googlesyndication', 'adriver', 'ok', 'icloud', 'buzzoola', '2mdn', 'betweendigital', 'googleapis', 'relap', 'otm-r', 'tinkoff', 'rambler', 'gosuslugi', 'amazonaws', 'rtb', 'mts', 'avito', 'playreplay', 't', 'facebook', 'weborama', 'sape', 'mycdn', 'smi2', 'pinimg', 'lenta', 'bidvol', 'moevideo', 'sbrf', 'aliexpress', 'skwstat', 'vtb', 'duckduckgo', 'thesame', 'samsungapps', 'realsrv', 'twitter', 'hh']

sites_with_different_audience = list(np.unique(sites_with_different_audience))

# @prepare_feature
# def share_of_sites_with_bias_in_audience_float32(dfi, sites_with_different_audience=sites_with_different_audience):
#     if isinstance(dfi, DFD):
#         return check_df_is_appropriate_for_dfis_and_insert_it(dfi, data_from_url_simple())
#     temp = dfi[2].value_counts() / dfi.shape[0]
#     res = {f"{site}_share": temp.loc[site] if site in temp.index else 0 for site in sites_with_different_audience}
#     return res

# @prepare_feature
# def share_of_sites_by_day_part_with_bias_in_audience_float32(dfi, sites_with_different_audience=sites_with_different_audience):
#     if isinstance(dfi, DFD):
#         return check_df_is_appropriate_for_dfis_and_insert_it(dfi,
#                         pd.concat([data_from_url_simple(),
#                                    get_new_features_from_part_of_day(load_col('part_of_day')),
#                                    ], axis=1))
#     for pod, data in dfi.groupby('part_of_day')[2].agg(list).items():
#         temp = pd.Series(data, dtype='object').value_counts() / len(data)
#         res = {f"{site}_share_{pod}": temp.loc[site] if site in temp.index else 0 for site in sites_with_different_audience}
#     return res

# @prepare_feature
# def phone_data_category(dfi):
#     if isinstance(dfi, DFD):
#         return check_df_is_appropriate_for_dfis_and_insert_it(dfi, process_manufacturer(load_col('cpe_manufacturer_name')))
#     res = dict()
#     res['phone_factory'] = dfi['cpe_manufacturer_name'].value_counts().index[0]
#     # res['phone_type'] = dfi['cpe_type_cd'].value_counts().index[0]
#     return res

@prepare_feature
def region_category(dfi):
    if isinstance(dfi, DFD):
        return check_df_is_appropriate_for_dfis_and_insert_it(dfi, simple_process_category(load_col('region_name')))
    res = dict()
    temp = dfi['region_name'].value_counts()
    res['region_count'] = len(dfi['region_name'].unique())
    res['region1'] = temp.index[0]
    return res


@prepare_feature
def add_url_data(dfi):
    if isinstance(dfi, DFD):
        return check_df_is_appropriate_for_dfis_and_insert_it(dfi, data_from_url())
    if 'turbopages' in dfi:
        dfi = dfi.drop(columns='turbopages')
    if 'is_small' in dfi:
        dfi['is_small'] = dfi['is_small'].fillna(False).astype(int)
    dfi = pd.get_dummies(dfi)
    dfi = dfi.mean(numeric_only=False)
    dfi = dfi.to_dict()
    return dfi
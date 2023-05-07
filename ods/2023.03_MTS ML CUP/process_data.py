from requests import get, Response
import time

import numpy as np
import pandas as pd
from tqdm import tqdm

from settings import path
from load import load, write


def dataframe_group_to_dict_by_user(file_name, df_load_fun, force=True,
                                    *args, **kwargs):
    dfis = load(file_name, dict())
    if force or len(dfis) == 0:
        print('start df splitting by users')
        df = df_load_fun()
        r = df['user_id'].diff()
        vals = r[r != 0].index
        dfis = {df.loc[v1, 'user_id']: df.iloc[v1:v2-1].drop(columns=['user_id']) for v1, v2 in zip(vals, vals[1:])}
        write(file_name, dfis)
    return dfis


class DFD():
    def __init__(self, d, df=None):
        self.d = d
        self.df = df
    
    def __getitem__(self, index):
        if self.df is None:
            raise ValueError('dataframe is not defined')
        ind = self.d[index]
        return self.df.iloc[ind[0]:ind[1]]
    
    def keys(self):
        return self.d.keys()
    

def dataframe_group_indexes_to_dict_by_user(file_name, df_load_fun, df=None):
    dfd = load(file_name)
    if df is None:
        df = df_load_fun()
    if dfd is None:
        if not np.all(df['user_id'].sort_values().values == df['user_id'].values):
            raise ValueError('dataframe must be sorted by user_id')
        r = df['user_id'].diff()
        vals = r[r != 0].index
        dfd = DFD({df.loc[v1, 'user_id']: (v1, v2) for v1, v2 in zip(vals, vals[1:])})
        write(file_name, dfd)
    dfd.df = df
    return dfd


def simple_process_category(data_col):
    return simple_process(data_col).astype('category')


def simple_process(data_col):
    return data_col if isinstance(data_col, pd.DataFrame) else data_col.to_frame()


def get_new_features_from_date(data_col, dayofweek=True, ):
    data_col = data_col.astype(np.datetime64)
    new_df = pd.DataFrame()
    if dayofweek:
        new_df['dow'] = (data_col.dt.dayofweek).astype('category')
    
    return new_df


def get_new_features_from_part_of_day(data_col):
    data_col = data_col.astype('category')
    new_df = pd.DataFrame(data_col)
    return new_df


def process_manufacturer(data_col):
    data_col = data_col.replace({'Realme Mobile Telecommunications (Shenzhen) Co Ltd': 'Realme',
                                 'Realme Chongqing Mobile Telecommunications Corp Ltd': 'Realme',
                                 'Highscreen Limited': 'Highscreen',
                                 'Huawei Device Company Limited': 'Huawei',
                                 'Motorola Mobility LLC, a Lenovo Company': 'Motorola',
                                 'Sony Mobile Communications Inc.': 'Sony',})
    
    return data_col.astype('category')


def process_url_n_set_to_data(data_col, url_cols=[0, 1, 2], *args, **kwargs):
    dfurl = data_col.copy()
    df = process_url(dfurl)
    df1 = df.drop(columns=[x for x in df.columns if not isinstance(x, str) and x not in url_cols]).set_index(0)
    for col in df1:
        df1[col] = df1[col].astype('category')
    
    df2 = df1.loc[dfurl]
    df2.index = dfurl.index
    return df2


def process_url_n_set_to_data_full(data_col, *args, **kwargs):
    dfurl = data_col.copy()
    df = process_url(dfurl)
    df1 = df.drop(columns=[x for x in df.columns if not isinstance(x, str) and x not in [0, ]]).set_index(0)
    for col in df1:
        df1[col] = df1[col].astype('category')
    df1 = df1.reset_index()
    
    df_add = get_additional_data_about_url(dfurl, allow_load=False, force_processing=True, verbose=0)
    cols = ['top_country_1', 'top_country_1_share', 'top_country_2', 'top_country_2_share', 'visits',
                         'country_rank_c', 'country_rank_r', 'traffic_source_social', 'traffic_source_paid_referrals',
                         'traffic_source_mail', 'traffic_source_referrals', 'is_small', 'turbopages',
                                  'cat_rank', 'global_rank', 'traffic_source_direct', 'traffic_source_search']
    df_add = df_add.drop(columns=[x for x in cols if x in df_add])
    
    df1 = df1.merge(df_add, on=0, how='left')
    df1 = df1.set_index(0)
    
    if True:
        # crashes in wsl
        df2 = df1.loc[dfurl]
    else:
        # also crashes in wsl
        print('merge start')
        step = int(1e6)
        dfurl = dfurl.to_frame()
        df1 = df1.reset_index()
        kwargs = {'left_on':'url_host', 'right_on':0, 'how': 'left'}
        l = dfurl.shape[0]
        r = []
        for i in range(int(np.ceil(l / step))):
            # # print(i, i * step, min((i + 1) * step, l))
            r.append(dfurl.iloc[i * step:min((i + 1) * step, l)].merge(df1, **kwargs).drop(columns=['url_host', 0]))
        df2 = pd.concat(r)
    
    df2.index = dfurl.index
    return df2


def origin_url_processing(data_col):
    cats = list(data_col.cat.categories)
    df = pd.DataFrame([[x] + list(reversed(x.split('.'))) for x in cats])
    cols = df.columns
    
    # df['narod'] = (df[2] == 'narod').astype('int32')
    # df['ucoz'] = (df[2] == 'ucoz').astype('int32')
    # df['gov'] = ((df[1] == 'gov') | (df[2] == 'gov') | (df[3] == 'gov')).astype('int32')
    df['turbopages'] = ((df[2] == 'turbopages') & (~df[3].isna())).astype('int32')
    
    switcher = (((df[2] == 'turbopages') & (~df[3].isna())).astype('int32')) == 1
    # switcher = (((df[0].str.contains('turbopages.org')) & (~df[3].isna())).astype('int32')) == 1
    new_df = df[switcher][3].str.replace('--', '$special_char$', regex=False).str.replace('-', '.', regex=False).str.replace('$special_char$', '-', regex=False).str.split('.')
    new_df = pd.DataFrame([reversed(x) if isinstance(x, list) else [x] for x in new_df], index=new_df.index)
    new_df = new_df.dropna(how='all')
    for col in new_df:
        df[col + 1] = df[col + 1].where(~switcher, new_df[col])
    
    switcher = (df[1] == 'org') & (df[2] == 'ampproject') & (df[3] == 'cdn')
    new_df = df[switcher][4].str.replace('--', '$special_char$', regex=False).str.replace('-', '.', regex=False).str.replace('$special_char$', '-', regex=False).str.split('.')
    new_df = new_df.apply(lambda x: x[1:-1] if x[0] == x[-1] == '0' else x)
    new_df = pd.DataFrame([reversed(x) if isinstance(x, list) else [x] for x in new_df], index=new_df.index)
    new_df = new_df.dropna(how='all')
    for col in new_df:
        df[col + 1] = df[col + 1].where(~switcher, new_df[col])
    return df


def get_additional_data_about_url(data_col, allow_load=False, pause=1, batch_size=10, verbose=1, force_processing=False):
    file_name = path['data']['outer_data'] / 'additional_url_data.pickle'
    file_name_processed = path['data']['processed_data'] / 'additional_url_data.pickle'
    if file_name_processed.exists() and not force_processing:
        return load(file_name_processed)
    data = load(file_name, dict())
    def get_additional_data(site):
        if not site:
            return site
        request = r"https://data.similarweb.com/api/v1/data?domain=" + site
        h = {'authority': 'data.similarweb.com',
            'method': 'GET',
            'path': '/api/v1/data?domain=0-34.ru',
            'scheme': 'https',
            'accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7',
            'accept-encoding': 'gzip, deflate, br',
            'accept-language': 'ru,en;q=0.9',
            'cache-control': 'max-age=0',
            'referer': 'http://127.0.0.1:8082/',
            'sec-ch-ua': '"Chromium";v="110", "Not A(Brand";v="24", "Google Chrome";v="110"',
            'sec-ch-ua-mobile': '?0',
            'sec-ch-ua-platform': 'Windows',
            'sec-fetch-dest': 'document',
            'sec-fetch-mode': 'navigate',
            'sec-fetch-site': 'cross-site',
            'sec-fetch-user': '?1',
            'upgrade-insecure-requests': '1',
            'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/110.0.0.0 Safari/537.36'}
        res = get(request, headers=h)
        return res.json() if res.status_code == 200 else res
    
    df = origin_url_processing(data_col)
    f = df[1].unique()
    def temp(row):
        if row[3] in f and row[4] is not None:
            return f"{row[4]}.{row[3]}.{row[2]}.{row[1]}"
        elif row[3] is not None:
            return f"{row[3]}.{row[2]}.{row[1]}"
        elif row[2] is not None:
            return f"{row[2]}.{row[1]}"
        else:
            return None
    df['sites'] = df.apply(temp, axis=1)
    def temp(row):
        if row[3] in f and row[4] is not None:
            return f"{row[4]}.{row[3]}.{row[2]}.{row[1]}"
        elif row[3] is not None and row[2] in f:
            return f"{row[3]}.{row[2]}.{row[1]}"
        elif row[2] is not None:
            return f"{row[2]}.{row[1]}"
        else:
            return None
    df['sites2'] = df.apply(temp, axis=1)
    
    s1 = df.merge(data_col.value_counts().to_frame().reset_index(), left_on=0, right_on='index', how='left').groupby('sites')['url_host'].sum().sort_values(ascending=False)
    s2 = df.merge(data_col.value_counts().to_frame().reset_index(), left_on=0, right_on='index', how='left').groupby('sites2')['url_host'].sum().sort_values(ascending=False)
    sites_count = pd.concat([s1.loc[list(set(s1.index) - set(s2.index))], s2]).sort_values(ascending=False)
    sites = sites_count.index
    
    sites = [x for x in sites if x not in list(data.keys()) or (x in list(data.keys()) and isinstance(data[x], Response) and data[x].status_code == 403)]
    if verbose > 0:
        print(f"site count: {len(sites)}")
        print(f"processed site count: {len(data)}")
    if allow_load:
        for i, site in (enumerate(sites)):
            if verbose == 1:
                print(i, end=', ')
            elif verbose == 2:
                print(f"{i} {site} ({sites_count.loc[site]})")
            if i % batch_size == 0 and i > 0:
                if verbose > 0:
                    print('save')
                write(file_name, data)
            data[site] = get_additional_data(site)
            if not isinstance(data[site], dict) and verbose > 0:
                print(f"{site}: {'OK' if isinstance(data[site], dict) else data[site]}")
            time.sleep(pause)
        write(file_name, data)
    
    res = []
    for v in data.values():
        if not isinstance(v, dict):
            continue
        cd = {'sites': v['SiteName']}
        for i in range(min(2, len(v['TopCountryShares']))):
            cd[f'top_country_{i + 1}'] = str(v['TopCountryShares'][i]['Country'])
            cd[f'top_country_{i + 1}_share'] = v['TopCountryShares'][i]['Value']
        cd['bounce_rate'] = v['Engagments']['BounceRate']
        cd['page_pre_visit'] = v['Engagments']['PagePerVisit']
        cd['visits'] = v['Engagments']['Visits']
        cd['time_on_site'] = v['Engagments']['TimeOnSite']
        cd['global_rank'] = v['GlobalRank']['Rank']
        cd['country_rank_c'] = str(v['CountryRank']['Country'])
        cd['country_rank_r'] = v['CountryRank']['Rank']
        cd['is_small'] = v['IsSmall']
        for col in v['TrafficSources']:
            cd[f"traffic_source_{col.replace(' ', '_').lower()}"] = v['TrafficSources'][col]
        if v['Category']:
            # for i, cat in enumerate(v['Category'].split('/')):
            #     cd[f'cat_{i + 1}'] = cat
            cd['cat'] = v['Category']
        cd['cat_rank'] = v['CategoryRank']['Rank']
        res.append(cd)
    res = pd.DataFrame(res)
    for col in ('top_country_1_share', 'top_country_2_share',
                'bounce_rate', 'page_pre_visit', 'global_rank',
                'country_rank_r', 'traffic_source_social',
                'traffic_source_paid_referrals',
                'traffic_source_mail','traffic_source_referrals',
                'traffic_source_search', 'traffic_source_direct',
                'visits', 'time_on_site', 'cat_rank'):
        if col in res:
            res[col] = res[col].astype('float32')
    for col in ('top_country_1', 'top_country_2',
                'country_rank_c', 'is_small', 'cat', 'cat_1', 'cat_2'):
        if col in res:
            res[col] = res[col].astype('category')
    res = res.drop_duplicates()

    x = df.merge(res, left_on='sites2', right_on='sites', how='left').merge(res, left_on='sites_x', right_on='sites', how='left')
    cols = [x for x in res if x not in ('sites', )]
    for col in cols:
        x[col] = x[col + '_y'].where(~x[col + '_y'].isna(), x[col + '_x'])
    x = x.drop(columns=([col + '_y' for col in cols]
                        + [col + '_x' for col in cols]
                        + [col for col in range(1, 10) if col in x]
                        + [col for col in ('narod', 'ucoz', 'turbopages', 'gov') if col in x]
                        + [col for col in x if isinstance(col, str) and col.startswith('sites')]))
    def temp(x, num=0):
        if isinstance(x, str):
            xl = x.split('/')
            if len(xl) >= (num + 1):
                return xl[num]
        return x[-1]
    x['cat_1'] = x['cat'].apply(temp).astype('category')
    x['cat_2'] = x['cat'].apply(lambda y: temp(y, 1)).astype('category')
    x = x.drop(columns='cat')
    
    x = x.set_index(0).dropna(how='all').reset_index()
    x = x.drop_duplicates()
    write(file_name_processed, x)
    return x
        

def process_url(data_col, replace_by_rare_deep={1: 10, 2: None}):
    if isinstance(data_col, pd.DataFrame):
        if data_col.shape[1] > 1:
            raise ValueError('data_col must contain one column')
        data_col = data_col.iloc[:, 0]
    else:
        data_col = data_col.copy()

    df = origin_url_processing(data_col)
    cols = [x for x in df if not isinstance(x, str)]

    to_left = ['ru', 'net', 'com', 'do', 'moy', 'co', 'org', 'nn', 'in', 'msk', 'spb', 'gov' ,'en', 'su']

    # to_left = []
    # switcher = ~df[2].isin(to_left)
    # df[1] = df[1].where(switcher, df[2].str.cat(df[1], sep='.'))
    # for col1, col2 in zip(cols[2:], cols[3:]):
    #     df[col1] = df[col1].where(switcher, df[col2])

#     to_left = ['ru', 'ua', 'su', 'in', 'co', 'spravka']
#     switcher = ~df[2].isin(to_left)
#     df[1] = df[1].where(switcher, df[2])
#     for col1, col2 in zip(cols[2:], cols[3:]):
#         df[col1] = df[col1].where(switcher, df[col2])

#     to_left = ['ru', 'net', 'com', 'do', 'moy', 'co', 'org', 'nn', 'in', 'msk', 'spb', 'gov' ,'en', 'su', 'narod', 'ucoz']
#     switcher = ~df[2].isin(to_left)
#     for col1, col2 in zip(cols[2:], cols[3:]):
#         df[col1] = df[col1].where(switcher, df[col2])
        
    def replace_by_rare(col, val):
        t = col.value_counts()
        return col.apply(lambda x: 'rare' if not pd.isna(x) and t.loc[x] < val else x)
    
    # replace = dict()
    # replace |= {x: 'g1' for x in ['eu', 'lv', 'de', 'pl', 'ge', 'lt', 'ee', 'fr', 'it', 'us', 'es', 'ie', 'lu',
    #                               'im', 'is', 'cz', 'vg', 'bg', 'ca', 'cat', 'ch', 'uk',' re', 'be', 'sg', 'ro',
    #                               'nl', 'tw', 'fi', 'gr', 'kr', 'jp', 'au']}
    # replace |= {x: 'g2' for x in ['pw', 'rs', 'gy', 'cx', 'tr', 'asia', 'cn', 'ws', 'to', 'hk', 'bz', 'io', 'ir', 'tk', 'in']}
    # replace |= {x: 'g3' for x in ['am', 'az']}
    # replace |= {x: 'g4' for x in ['uz', 'tj', 'kg', 'md']}
    # replace |= {x: 'fm' for x in ['fm', 'radio']}
    # df[1] = df[1].replace(replace)

    uncorrect_domens = [x for x in df[1].unique() if x.isnumeric()] + ['-1', '_', ' ', '']
    df[1] = df[1].replace({x: '-1' for x in uncorrect_domens})
    # df[1] = df[1].apply(lambda x: 'xn-' if x.startswith('xn-') else x)
        
    # replace = dict()
    # replace |= {x: 'job' for x in ['hh', 'superjob', 'jobfilter', 'gorodrabot', 'gorjob', 'jobcareer']}
    # df[2] = df[2].replace(replace)

    # replace = dict()
    # replace |= {x: 'sprav' for x in ['sprav', ]}
    # replace |= {x: 'film' for x in ['film', 'kino', 'cinema', ]}
    # replace |= {x: 'auto' for x in ['auto', 'avto', 'koles', 'zarul', ]}
    # for key, val in replace.items():
    #     df[2] = df[2].where(~df[2].str.contains(key, na=False), val)

    # df[2] = df[2].apply(lambda x: 'xn-' if not pd.isna(x) and x.startswith('xn-') else x)
    
    for k, v in replace_by_rare_deep.items():
        if v is not None:
            df[k] = replace_by_rare(df[k], replace_by_rare_deep[k])
    return df




# def dataframe_group_to_dict_by_user(file_name, df_load_fun, force=False,
#                                     time_for_step_in_minutes=5, step_size=500):
    # dfis = load(file_name, dict())
    # if force or len(dfis) == 0:
    #     df = df_load_fun()
    #     df = df.sort_values('user_id')
    #     print('length of dfis:', len(dfis))
    #     user_ids = set(df['user_id'].unique()) - set(dfis.keys())
    #     print('working with df')
    #     df = df[df['user_id'].isin(user_ids)]
    #     print('working with dfis')
    #     s = time.time()
    #     for i, user_id in tqdm(list(enumerate(user_ids, 1))):
    #         dfis[user_id] = df[df['user_id'] == user_id]
    #         if i % step_size == 0:
    #             delta = time.time() - s
    #             if delta > 60 * time_for_step_in_minutes:
    #                 print(f"save on {i} step ({delta:.1f} seconds)")
    #                 s = time.time()
    #                 write(file_name, dfis)
    #                 print(f"write on disk for {time.time() - s:.1f} seconds")
    #                 df = df[df['user_id'].isin(set(df['user_id'].unique()) - set(dfis.keys()))]
    #                 s = time.time()
    # return dfis
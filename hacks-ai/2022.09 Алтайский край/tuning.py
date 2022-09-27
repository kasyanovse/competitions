from copy import deepcopy
from multiprocessing import Pool
import functools

import pandas as pd
from tqdm.notebook import tqdm

from parameters import THREADS


def run_fun(fun, datas, **kwargs):
    fun = functools.partial(fun, **kwargs)
    if 'parallel' in kwargs and kwargs['parallel']:
        with Pool(THREADS) as p:
            return p.map(fun, tqdm(datas))
    else:
        return [fun(x) for x in tqdm(datas)]


def try_each_col(fun, data, **kwargs):
    f1, t1, f2, t2 = deepcopy(data)
    datas = [(f1[[col]], t1, f2[[col]], t2) for col in data[0].columns]
    r = run_fun(fun, datas, **kwargs)
    return pd.DataFrame([{'col': col} | res for col, res in zip(data[0].columns, r)])


def try_wo_each_col(fun, data, **kwargs):
    f1, t1, f2, t2 = deepcopy(data)
    datas = [(f1.drop(columns=[col]), t1, f2.drop(columns=[col]), t2) for col in data[0].columns] + [(f1, t1, f2, t2)]
    r = run_fun(fun, datas, **kwargs)
    return pd.DataFrame([{'col': col} | res[1] for col, res in zip(data[0].columns, r)]
                        + [{'col': None} | r[-1][1]])


def try_sets_of_cols(fun, data, cols_sets, **kwargs):
    f1, t1, f2, t2 = deepcopy(data)
    datas = [(f1[cols], t1, f2[cols], t2) for cols in cols_sets]
    r = run_fun(fun, datas, **kwargs)
    if 'only_res' in kwargs and kwargs['only_res']:
        r = pd.DataFrame([{'num': i} | res for i, (col, res) in enumerate(zip(data[0].columns, r))])
    else:
        r = pd.DataFrame([{'num': i} | res | {'model': model} for i, (col, (model, res)) in enumerate(zip(data[0].columns, r))])
    r['delta'] = (r['test'] - r.iloc[-1]['test'])
    return r


def try_cols_in_order(fun, data, cols, **kwargs):
    f1, t1, f2, t2 = deepcopy(data)
    cols = [x for x in cols if x in f1.columns]
    cols_sets = [[x for j, x in enumerate(cols) if j <= i] for i in range(len(cols))]
    datas = [(f1[cols_set], t1, f2[cols_set], t2)for cols_set in cols_sets]
    r = run_fun(fun, datas, **kwargs)
    return pd.DataFrame([{'i': i} | res for i, res in enumerate(r)])
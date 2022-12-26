""" funs for working with features 
    first letter of funs:
        t - transformer: get data and transform it
        f - feature creator: get data and return data with new features """

import math
from datetime import datetime

import numpy as np
from scipy.signal import TransferFunction, lsim
import pandas as pd
from sklearn.preprocessing import PolynomialFeatures
from sklearn.decomposition import PCA, KernelPCA, FastICA, FactorAnalysis

from utils import TARGET_FEATURE, target_split, read, write, load_all

pd.set_option('mode.chained_assignment', None)


""" decorators """
def apply_nf(fun):
    """ split dataframe to features and target before appliyng fun and concatenate after """
    def apply_nf_temp(df, *args, **kwargs):
        is_series = isinstance(df, pd.Series)
        if is_series:
            df = df.to_frame()
        switch = isinstance(df, pd.DataFrame) and TARGET_FEATURE in df.columns
        if switch:
            df, t = target_split(df)
        df = fun(df, *args, **kwargs)
        if switch:
            df = pd.concat([df, t.loc[df.index]], axis=1)
        if is_series:
            df = df.iloc[:, 0]
        return df
    apply_nf_temp.__name__ = fun.__name__
    return apply_nf_temp


""" main funs """

def t_fillna(df, filler=0):
    return df.fillna(filler)

def t_delna(df):
    return df.dropna()

""" working with time """
def is_datetime_index(df):
    types = (np.datetime64, pd.Timestamp, datetime)
    return isinstance(df.index[0], types)


def get_time(df):
    if is_datetime_index(df):
        t = pd.Series(df.index).astype(np.int64) * 1e-9
    else:
        t = pd.Series(df.index)
    t -= t.iloc[0]
    return t


def get_time_step(df, raise_error_on_multiple_value=True, filter_fun=lambda x: x[0]):
    dt = get_time(df).diff().dropna().unique()
    if raise_error_on_multiple_value and len(dt) > 1:
        raise ValueError(f'there are some different time steps: {dt}')
    return filter_fun(dt)


def set_time_in_sec(df):
    df.index = get_time(df)
    return df


def resampling(df, dt):
    time = get_time(df)
    df.index = time
    new_time = np.arange(time.iloc[0], time.iloc[-1], dt)
    new_time = np.setdiff1d(new_time, time)
    df = pd.concat([df, pd.DataFrame(index=new_time)]).sort_index()
    df = df.interpolate()
    return df


""" transformer functions """
@apply_nf
def t_discrete(x, num=100, drop_duplicates=False):
    x = ((x - x.min()) / (x.max() - x.min()) * num).round()
    x = x.dropna(axis=0, how='all')
    x = x.dropna(axis=1, how='all')
    return x.drop_duplicates() if drop_duplicates else x


@apply_nf
def t_dropna(df):
    return df.dropna()


@apply_nf
def t_stationary(df):
    return df.diff()


@apply_nf
def t_q_q(df, q=(0.1, 0.9)):
    def qq(series, q):
        qa1, qa2 = series.quantile(q[0]), series.quantile(q[1])
        return qa1, qa2, qa2 - qa1
    
    for col in df.columns:
        if len(df[col].unique()) < 10:
            continue
        q1 = q
        qa1, qa2, d = qq(df[col], q1)
        while d == 0:
            q1 = (q1[0] - q1[0] * 0.5, q1[1] + q1[0] * 0.5)
            qa1, qa2, d = qq(df[col], q1)
        df[col] = (df[col] - qa1) / d
    return df


@apply_nf
def t_to_positive(df):
    return df - df.min()


@apply_nf
def t_min_max(df):
    for col in df.columns:
        if df[col].max() != df[col].min():
            df[col] = (df[col] - df[col].min()) / (df[col].max() - df[col].min())
        else:
            df[col] = (df[col] - df[col].min())
    return df


@apply_nf
def t_limit(df, edges=(-2, 2)):
    df[df > edges[1]] = edges[1]
    df[df < edges[1]] = edges[0]
    return df


@apply_nf
def t_tf(df, num, den):
    index = df.index
    df = set_time_in_sec(df)
    tf = TransferFunction(num, den)
    for col in df.columns:
        df[col] = lsim(tf, U=df[col], T=df.index)[1]
    df.index = index
    return df


@apply_nf
def t_mean_window_with_db(df, window, db):
    edges = (df.max() - df.min()) * db
    means = df.rolling(window=window).mean()
    means = means.fillna(method='bfill')
    means_low = means - edges
    means_high = means + edges
    df[(df > means_low) & (df < means_high)] = means
    df[(df < means_low)] += edges
    df[(df > means_high)] -= edges
    return df


@apply_nf
def t_mean_window60_with_db05(df):
    return t_mean_window_with_db(df, 60, 0.05)


@apply_nf
def t_mean_window30_with_db05(df):
    return t_mean_window_with_db(df, 30, 0.05)


def get_impulse_response(dt, T, amp=1e3, max_time_coeff=3):
    """ additional func for fast filtering
        dump library with results into file """
    file_name = 'dump/imp_resp_base.pickle'
    base = read(file_name, dict())
    pars = (dt, T, amp, max_time_coeff)
    if pars not in base:
        t = np.arange(0, T * max_time_coeff, dt)
        yin = t * 0
        yin[1] = amp
        res = lsim([(T, 0), (T, 1)], U=yin, T=t)[1]
        res[1] -= sum(res)
        base[pars] = res[1:] / (amp * dt)
        write(file_name, base)
    return base[pars]


""" new features functions """


## add means functions
@apply_nf
def f_window_mean_diff(df, window):
    means = df.rolling(window=window).mean() - df
    means = means.fillna(method='bfill')
    means.columns = [f"{x}_mean{window}_diff" for x in df.columns]
    return pd.concat([df, means], axis=1)


@apply_nf
def f_window_mean05_diff(df):
    return f_window_mean_diff(df, 5)


@apply_nf
def f_window_mean10_diff(df):
    return f_window_mean_diff(df, 10)


@apply_nf
def f_window_mean20_diff(df):
    return f_window_mean_diff(df, 20)


@apply_nf
def f_window_mean30_diff(df):
    return f_window_mean_diff(df, 30)


@apply_nf
def f_window_mean50_diff(df):
    return f_window_mean_diff(df, 50)


@apply_nf
def f_window_mean60_diff(df):
    return f_window_mean_diff(df, 60)


@apply_nf
def f_window_mean75_diff(df):
    return f_window_mean_diff(df, 75)


@apply_nf
def f_window_mean100_diff(df):
    return f_window_mean_diff(df, 100)


@apply_nf
def f_window_mean(df, window):
    means = df.rolling(window=window).mean()
    means = means.fillna(method='bfill')
    means.columns = [f"{x}_mean{window}" for x in df.columns]
    return pd.concat([df, means], axis=1)


@apply_nf
def f_window_mean10(df):
    return f_window_mean(df, 10)


@apply_nf
def f_window_mean20(df):
    return f_window_mean(df, 20)


@apply_nf
def f_window_mean30(df):
    return f_window_mean(df, 30)


@apply_nf
def f_window_mean50(df):
    return f_window_mean(df, 50)


@apply_nf
def f_window_mean60(df):
    return f_window_mean(df, 60)


@apply_nf
def f_window_mean75(df):
    return f_window_mean(df, 75)


## polynomial features
@apply_nf
def f_poly_features(df, io=False, degree=2):
    pf = PolynomialFeatures(degree=degree, interaction_only=io, include_bias=False)
    pdf = pd.DataFrame(pf.fit_transform(df), index=df.index,
                       columns=pf.get_feature_names_out(df.columns))
    return pd.concat([df, pdf], axis=1)


@apply_nf
def f_poly_features_io(df):
    return f_poly_features(df, True)


@apply_nf
def f_poly_features3_io(df):
    return f_poly_features(df, False, 3)


@apply_nf
def f_poly_features3(df):
    return f_poly_features(df, True, 3)


## PCA
@apply_nf
def t_pca(df, n_components=10):
    model = PCA(n_components=n_components)
    df = pd.DataFrame(model.fit_transform(df), index=df.index)
    df.columns = map(str, range(df.shape[1]))
    return df


@apply_nf
def t_pca099(df):
    return t_pca(df, 0.99)


@apply_nf
def t_pca0999(df):
    return t_pca(df, 0.999)


@apply_nf
def t_pca10(df):
    return t_pca(df, 10)


@apply_nf
def t_pca15(df):
    return t_pca(df, 15)


@apply_nf
def t_pca5(df):
    return t_pca(df, 5)


@apply_nf
def t_ica(df, n_components=10):
    model = FastICA(n_components=n_components, max_iter=2000)
    df = pd.DataFrame(model.fit_transform(df), index=df.index)
    df.columns = map(str, range(df.shape[1]))
    return df


@apply_nf
def t_ica10(df):
    return t_ica(df, 10)


@apply_nf
def t_ica15(df):
    return t_ica(df, 15)


@apply_nf
def t_ica5(df):
    return t_ica(df, 5)


@apply_nf
def t_fa(df, n_components=10):
    model = FactorAnalysis(n_components=n_components, max_iter=2000)
    df = pd.DataFrame(model.fit_transform(df), index=df.index)
    df.columns = map(str, range(df.shape[1]))
    return df


@apply_nf
def t_fa10(df):
    return t_fa(df, 10)


@apply_nf
def t_fa15(df):
    return t_fa(df, 15)


@apply_nf
def t_fa5(df):
    return t_fa(df, 5)


## derivative functions
def f_derivative_base(df, ddt=1, dtt=2):
    dt = get_time_step(df)
    res = get_impulse_response(dt*ddt, dt*dtt*ddt)
    der = df.apply(lambda x: np.convolve(x.values.reshape(-1), res, 'same'))
    der = der.fillna(method='bfill')
    der.columns = [f"{x}_der" for x in df.columns]
    return der


@apply_nf
def f_derivative(df):
    return pd.concat([df, f_derivative_base(df)], axis=1)


@apply_nf
def f_derivative20(df):
    return pd.concat([df, f_derivative_base(df, 1, 20)], axis=1)


@apply_nf
def f_derivative10_2(df):
    return pd.concat([df, f_derivative_base(df, 2, 10)], axis=1)


@apply_nf
def f_derivative10_3(df):
    return pd.concat([df, f_derivative_base(df, 3, 10)], axis=1)


@apply_nf
def f_derivative10_4(df):
    return pd.concat([df, f_derivative_base(df, 4, 10)], axis=1)


@apply_nf
def f_derivative4_8(df):
    return pd.concat([df, f_derivative_base(df, 8, 4)], axis=1)


@apply_nf
def f_derivative_with_db(df):
    der = f_derivative_base(df)
    edge = (der.max() - der.min()) * 0.05
    der[der.abs() < edge] = 0
    der[der > edge] -= edge
    der[der < edge] += edge
    return pd.concat([df, der], axis=1)


@apply_nf
def f_two_sided_derivative(df):
    der1 = f_derivative_base(df)
    der2 = f_derivative_base(pd.DataFrame(df.values[::-1], index=df.index, columns=df.columns))
    der = (der1 + der2) / 2
    return pd.concat([df, der], axis=1)


@apply_nf
def f_diff(df):
    diff = df.shift(1) - df
    diff = diff.fillna(method='bfill')
    diff.columns = [f"{x}_diff" for x in diff.columns]
    diff['abs_diff'] = (diff ** 2).sum(axis=1) ** 0.5
    return pd.concat([df, diff], axis=1)


@apply_nf
def f_diff2(df):
    diff = df.diff().rolling(window=2).mean()
    diff = diff.fillna(method='bfill')
    diff.columns = [f"{x}_diff" for x in diff.columns]
    diff['abs_diff'] = (diff ** 2).sum(axis=1) ** 0.5
    return pd.concat([df, diff], axis=1)


@apply_nf
def f_diff3(df):
    diff = df.diff().rolling(window=3).mean()
    diff = diff.fillna(method='bfill')
    diff.columns = [f"{x}_diff" for x in diff.columns]
    diff['abs_diff'] = (diff ** 2).sum(axis=1) ** 0.5
    return pd.concat([df, diff], axis=1)


@apply_nf
def f_diff4(df):
    diff = df.diff().rolling(window=4).mean()
    diff = diff.fillna(method='bfill')
    diff.columns = [f"{x}_diff" for x in diff.columns]
    diff['abs_diff'] = (diff ** 2).sum(axis=1) ** 0.5
    return pd.concat([df, diff], axis=1)


@apply_nf
def f_diff10(df):
    diff = df.diff().rolling(window=10).mean()
    diff = diff.fillna(method='bfill')
    diff.columns = [f"{x}_diff" for x in diff.columns]
    diff['abs_diff'] = (diff ** 2).sum(axis=1) ** 0.5
    return pd.concat([df, diff], axis=1)


## correlation deletion functions
@apply_nf
def del_corr(df, max_corr=0.7):
    cols = df.columns
    to_skip = []
    for i1 in range(len(cols) - 1):
        if cols[i1] in to_skip:
                continue
        for i2 in range(i1 + 1, len(cols)):
            if cols[i2] in to_skip:
                continue
            if abs(df[cols[[i1, i2]]].corr().iloc[0, 1]) > max_corr:
                to_skip.append(cols[i2])
    return df.drop(columns=to_skip)


@apply_nf
def replace_corr(df, max_corr=0.7, normalize=False):
    cols = df.columns
    to_skip = []
    groups = []
    for i1 in range(len(cols) - 1):
        groups.append([cols[i1]])
        if cols[i1] in to_skip:
                continue
        for i2 in range(i1 + 1, len(cols)):
            if cols[i2] in to_skip:
                continue
            if abs(df[cols[[i1, i2]]].corr().iloc[0, 1]) > max_corr:
                to_skip.append(cols[i2])
                groups[-1].append(cols[i2])
    groups = [x for x in groups if len(x) > 1]
    for i, cols in enumerate(groups, 1):
        if normalize:
            df[f'new_{i}'] = t_min_max(df[cols]).sum(axis=1)
        else:
            df[f'new_{i}'] = df[cols].sum(axis=1)
    return df.drop(columns=to_skip)


## add new combination of features
@apply_nf
def f_abs(df):
    df['abs'] = (df.abs()).sum(axis=1)
    return df


@apply_nf
def f_abs2(df):
    df['abs2'] = (df ** 2).sum(axis=1) ** 0.5
    return df


@apply_nf
def f_abs05(df):
    df['abs05'] = (df.abs() ** 0.5).sum(axis=1) ** 2
    return df

""" series expand and oscillations"""

def oscillated_columns(df, portion_of_one_freq=0.01, edge=2.2, window=10):
    res = []
    import matplotlib.pyplot as plt
    from utils import FIGSIZE_NARROW as FIGSIZE
    for col in df.columns:
        f, w = fourie(df, col)
        
        _, ax = plt.subplots(figsize=FIGSIZE)
        plt.plot(f[1:], abs(w[1:]))
        ax.set_title(col)
        
        n = max([x for x, y in enumerate(abs(w) / max(abs(w[1:])) > portion_of_one_freq) if y])
        w = pd.Series(abs(w[1:n]) / max(abs(w[1:])), index=f[1:n])
        if any((w > w.rolling(window).max().rolling(window).mean() * edge) & (w > portion_of_one_freq)):
            res.append(col)
            
        # _, ax = plt.subplots(figsize=FIGSIZE)
        # w.plot(ax=ax)
        # w.rolling(window).max().rolling(window).mean().plot(ax=ax)
        # ax.set_title(col)
    return res

def fourie(df, col):
    if is_datetime_index(df):
        time = [x.timestamp() for x in pd.Series(df.index).dt.to_pydatetime()]
    else:
        time = df.index
    return fourie0(np.array(time), df[col].values)

def fourie0(time:np.array, signal:np.array, interp_count:int=None, sort_f:float=None):
    ''' expand signal to Fourie series
        :param interp_count: amount of points for interpolation
        :param sort_f: function for choosing frequences to output, get array of frequences, return bool array
    '''
    # interpolate
    if interp_count is None:
        interp_count = 10e3
    interp_count = int(interp_count)
    next_pow = (interp_count - 1).bit_length()
    interp_count = int(2 ** next_pow)
    delta = time[0]
    new_time = np.linspace(time[0], time[-1], interp_count)
    time, signal = new_time - delta, np.interp(new_time, time, signal)

    dt = time[1] - time[0]
    l = int(interp_count / 2 - 1)
    f = np.arange(l + 1) / (dt * interp_count)
    w = np.fft.fft(signal)[:l + 1] / interp_count
    w[1:] = 2 * np.abs(w[1:]) * np.exp(complex(0, 1) * (np.angle(w[1:]) - delta * f[1:] * 2 * math.pi))
    if sort_f is not None:
        f_b = sort_f(f)
        f, w = f[f_b], w[f_b]
    return f, w


def series_to_time(time, f, w):
    s = np.zeros(len(time))
    for _f, _w in zip(f, w):
        s += abs(_w) * np.cos(2 * math.pi * _f * time + np.angle(_w))
    return s

## testing
if __name__ == '__main__':
    """ evaluate baseline metrics variation due to different dataframe processing """    
    from math import ceil
    from itertools import product, chain
    from pathlib import Path
    from joblib import Parallel, delayed
    from tqdm import tqdm
    from scipy.stats import hmean
    from baseline import baseline_clusterization_score
    from scoring import CLUSTERIZATION_SCORERS_OLD, CLUSTERIZATION_SCORERS
    
    dump_file = Path('dump/feature_evaluating.pickle')
    db = read(dump_file, dict())
    
    dfs = load_all()
    
    def temp(tf, dfs=dfs):
        tf = [x for x in tf if x]
        res = {x.__name__: 1 for x in tf}
        for idf, df in enumerate([df.copy() for df in dfs], 1):
            for itf in tf:
                df = itf(df)
            res |= {f"{k}_{idf}": v for k, v in baseline_clusterization_score(df).items()}
        return res
        
    def tf_to_key(tf):
        return tuple(sorted([x.__name__ for x in tf if x]))
    
    groups = [(None, t_min_max, t_q_q, t_to_positive), (None, del_corr, replace_corr), (None, f_abs, f_abs2, f_abs05), (None, f_derivative, f_derivative_with_db, f_two_sided_derivative, f_diff),
              (None, f_window_mean50_diff, f_window_mean60_diff, f_window_mean75_diff, f_window_mean10, f_window_mean30, f_window_mean50, f_window_mean75),]
    groups = [(None, t_to_positive), (None, replace_corr), (None, f_abs, f_abs2), (None, f_diff2, f_diff3, f_diff4, f_derivative4_8, f_derivative10_4, f_derivative10_2, f_derivative20, f_derivative, f_two_sided_derivative, f_diff),
              (None, t_mean_window30_with_db05, t_mean_window60_with_db05, f_window_mean50_diff, f_window_mean60_diff, f_window_mean75_diff, f_window_mean30, f_window_mean50,),]
    groups = [(t_pca099, t_pca0999, t_pca5, t_pca10, t_ica5, t_ica10, t_ica15, t_fa5, t_fa10, t_fa15), (None, f_poly_features, f_poly_features_io), (None, f_abs, f_abs2), (None, f_derivative, f_diff10),
              (None, f_window_mean30_diff, f_window_mean50_diff, f_window_mean10, f_window_mean30, f_window_mean50),]
    groups = [(t_pca0999, ), (None, f_abs2), (None, f_derivative, f_diff10),
          (None, f_window_mean30_diff, f_window_mean50_diff, ),]
    # groups = [(None, t_pca099, t_pca0999, t_pca5, t_pca10, t_pca15), (None, f_abs, f_abs2), (None, f_derivative, f_diff),
              # (None, f_window_mean60_diff, f_window_mean50_diff, f_window_mean10, f_window_mean30, f_window_mean50),]
    
    full_tfs = list(product(*groups))
    tfs = [x for x in full_tfs if tf_to_key(x) not in db]
    # tfs = []
    print(f"there are {len(tfs)} cases")
    
    if True:
        for itf, tf in enumerate(tfs):
            print(itf, [x.__name__ for x in tf if x])
            try:
                if tf_to_key(tf) in db:
                    print('key exists')
                    breakpoint()
                db[tf_to_key(tf)] = temp(tf)
                write(dump_file, db)
            except:
                breakpoint()
    else:
        processes = 5
        batch_size = processes*2
        for batch in [tfs[batch_size*i:batch_size*(i+1)] for i in range(ceil(len(tfs) / batch_size))]:
            add_res = Parallel(n_jobs=processes, prefer='processes', max_nbytes='1000M')(delayed(temp)(tf) for tf in tqdm(batch))
            db |= {tf_to_key(tf): res for tf, res in zip(batch, add_res)}
            write(dump_file, db)
            print('saved')
    result = [res | ({'key': tf_name} if tf_name else {}) for tf_name, res in sorted(db.items(), key=lambda x: ''.join(x[0]))]

    print('delete unusefull metrics')
    unusefull_scores = np.setdiff1d(list(CLUSTERIZATION_SCORERS_OLD), list(CLUSTERIZATION_SCORERS)) 
    usefull_result_feature = lambda k: 'cs_' in k or 'hs_' in k
    is_unusefull = lambda k: any([k.startswith(x) for x in unusefull_scores])
    def retain_only_usefull_metrics(res):
        res_new = {k: v for k, v in res.items() if usefull_result_feature(k)}
        temp = lambda num: {f"m_{num}": hmean([v for k, v in res_new.items() if str(num) in k])}
        res = {k: v for k, v in res.items() if not is_unusefull(k)}
        return res | temp(1) | temp(2)
    result = list(map(retain_only_usefull_metrics, result))
    
    for i in range(1, len(result)):
        td = {f"d_{k}": (result[i][k] - result[0][k]) / result[0][k] for k in result[0]}
        result[i] |= td | {'sum': sum(td.values())}
    r = pd.DataFrame(result)
    first_cols = list(result[0]) + [f"d_{k}" for k in result[0]]
    r = r[first_cols + list(sorted([col for col in r if col not in first_cols]))]
    r.to_excel('feature_res.xlsx')
    
    r2 = pd.DataFrame()
    g = r.drop(columns=first_cols + ['key', 'sum']).fillna(0)
    eval_fun0 = lambda i: r.loc[i]['sum'].mean()
    eval_fun = lambda i: eval_fun0(np.setdiff1d(i, [0]))
    for col in g.columns:
        i1 = g[g[col] == 1].index
        i2 = g.drop(columns=col).duplicated(keep=False)
        i2 = np.setdiff1d(i2[i2].index, i1)
        r2.at[col, '1'] = eval_fun(i1)
        r2.at[col, '2'] = eval_fun(i2)
        r2.at[col, 'd'] = eval_fun(i1) - eval_fun(i2)
        print(f"{col:30s}  1:{eval_fun(i1):+.4f}  0:{eval_fun(i2):+.4f}  delta:{eval_fun(i1) - eval_fun(i2):.4f}")
    r2.to_excel('feature_best.xlsx')
            

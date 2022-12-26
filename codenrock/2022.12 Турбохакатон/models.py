#!/usr/bin/env python
# -*- coding: utf-8 -*-

""" module with some models for problem """

from itertools import takewhile, count, accumulate

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
from tqdm.notebook import tqdm

from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, ClusterMixin, clone
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.metrics import silhouette_score
from sklearn.ensemble import RandomForestClassifier

from utils import TARGET_FEATURE, RANDOM_SEED, N_JOBS, TRANSIENT_FEATURE, target_split, mode_sequencer
from features import t_pca, t_discrete, t_min_max, oscillated_columns
from scoring import classification_score, clusterization_score


class Equalizer():
    def __init__(self, *args, n=((1, 12), (4, 12)), **kwargs):
        print('eq init')
        self.n = n
        super().__init__(*args, **kwargs)

    def predict(self, x=None, y=None):
        print('eq predict')
        res = super().predict(x, y)
        return self.transform(y=res)

    def fit_predict(self, x=None, y=None):
        return self.predict(x, y)

    def transform(self, x=None, y=None):
        a = np.array(y)
        for n in range(*self.n[0]):
            for i in range(n, len(a) - (n * 2 - 1)):
                if all(a[i-n:i] == a[i+n:i+2*n]) and any(a[i-n:i] != a[i:i+n]):
                    a[i:i+n] = a[i-n:i]
        for n in range(*self.n[1]):
            for i in range(n, len(a) - (n * 2 - 1)):
                n1 = np.unique(a[i-n:i])
                n2, c2 = np.unique(a[i:i+n], return_counts=True)
                n3 = np.unique(a[i+n:i+2*n])
                if len(n2) != 1 and len(n1) == 1 and len(n3) == 1 and n1[0] != n3[0]:
                    mp = sorted(zip(n2, c2), key=lambda x: x[1])[-1][0]
                    a[i:i+n] = [x if x in (n1[0], n3[0]) else mp for x in a[i:i+n]]
        if isinstance(y, pd.DataFrame):
            y = pd.DataFrame(a, index=y.index, columns=y.columns)
        elif isinstance(y, pd.Series):
            y = pd.Series(a, index=y.index, name=y.name)
        else:
            y = a
        return y

    def fit_transform(self, x=None, y=None):
        return self.transform(x, y)


class FeatureExtractor():
    def __init__(self, model=None,
                 summary_best_feature_importance_edge_values=0.5,
                 feature_intersection_edge=2):
        if model is None:
            model = RandomForestClassifier(random_state=RANDOM_SEED, verbose=False)
        self.model = model
        self.edge = summary_best_feature_importance_edge_values
        self.feature_intersection_edge = feature_intersection_edge
    
    def extract_features(self, x, y, parallel=True):
        def get_features(model, x, y, mode):
            model = clone(model).fit(x, (y == mode).astype(int))
            return dict(zip(x.columns, model.feature_importances_))

        modes = sorted(y.unique())
        if parallel:
            features = Parallel(n_jobs=N_JOBS, prefer='processes')(delayed(get_features)(self.model, x, y, mode) for mode in tqdm(modes))
        else:
            features = [get_features(self.model, x, y, mode) for mode in tqdm(modes)]
        features = dict(zip(modes, features))
        
        def extract_main_features(data, edge=self.edge):
            filtered_data = dict()
            filtered_features = dict()
            for mode, features in data.items():
                features = sorted(features.items(), key=lambda x: x[1])[::-1]
                n = len(list(takewhile(lambda x: x < edge,
                                       accumulate(features,
                                                  lambda x, y: x + y[1],
                                                  initial=0))))
                filtered_data[mode] = dict(features[:n])
                for feature in filtered_data[mode]:
                    if feature not in filtered_features:
                        filtered_features[feature] = 0
                    filtered_features[feature] += 1
            max_feature = ([filtered_features[list(features)[0]] for features in filtered_data.values()])
            filtered_features = {k: v for k, v in filtered_features.items() if v <= 5 or v in max_feature}
            
            for mode in filtered_data:
                filtered_data[mode] = [k for k in filtered_data[mode] if k in filtered_features]
            return filtered_data
        features = extract_main_features(features)
        return features
    
    def glue_modes_by_main_features(self, x, y, features):
        def intersection(d1, d2):
            d1 = d1[:5]
            d2 = d2[:5]
            d0 = set(d1) & set(d2)
            return sum([1 / (d1.index(x) + 1) ** 0.5 for x in d0]) + sum([1 / (d2.index(x) + 1) ** 0.5 for x in d0])
        res = []
        modes = list(features)
        for i1 in range(len(modes) - 1):
            for i2 in range(i1, len(modes)):
                i = intersection(features[modes[i1]], features[modes[i2]])
                if i > self.feature_intersection_edge:
                    res.append([i1, i2])
        mode_groups = [res.pop()]
        while res:
            i1, i2 = res.pop()
            for group in mode_groups:
                if i1 in group or i2 in group:
                    if i1 not in group:
                        group.append(i1)
                    if i2 not in group:
                        group.append(i2)
            else:
                mode_groups.append([i1, i2])
        for i1 in range(len(mode_groups) - 1):
            for i2 in range(i1 + 1, len(mode_groups)):
                if any(np.intersect1d(mode_groups[i1], mode_groups[i2])):
                    mode_groups[i1].extend(mode_groups[i2])
                    mode_groups[i2].clear()
        mode_groups = [list(set(group)) for group in mode_groups if group]
        replacer = dict()
        for group in mode_groups:
            origin, *last = group
            replacer |= {k: origin for k in last}
        return y.replace(replacer)


class TransientFinder():
    def __init__(self, pca_var=0.999, discrete_steps=100, min_difference=1e-3,
                 max_portion_of_transient_modes=0.18):
        self.pca_var = pca_var
        self.discrete_steps = discrete_steps
        self.min_difference = min_difference
        self.max_portion_of_transient_modes = max_portion_of_transient_modes

    def process_df(self, x, y):
        return t_discrete(t_pca(x, self.pca_var), self.discrete_steps, False), y

    def calculate_diffs(self, x, y):
        df = pd.concat(self.process_df(x, y), axis=1)
        modes = [target_split(mode) for mode in mode_sequencer(df)]
        m_x = pd.concat([i[0].diff().mean() for i in modes], axis=1).T
        m_y = pd.Series([i[1].iloc[0] for i in modes], name=TARGET_FEATURE)
        return m_x, m_y

    def predict(self, x, y, use_edge=False):
        m_x, m_y = self.calculate_diffs(x, y)
        m_x = (m_x.abs() > self.discrete_steps * self.min_difference).sum(axis=1)
        res = pd.concat([m_x, m_y], axis=1).groupby(TARGET_FEATURE)[0].mean()
        res.name = TRANSIENT_FEATURE
        y = y.replace(t_min_max(res).to_dict())
        if use_edge:
            a = y.value_counts().sort_index(ascending=False)
            a = a.index[a.index < self.max_portion_of_transient_modes]
            edge = pd.Series(a, index=a).diff().abs().idxmax() + 1e-9
            transient = (y > edge).astype(int)
            
            # y_tr = y * transient
            # y_tr = y_tr[y_tr != 0]
            # ydiff = y_tr.diff()
            # ind = ydiff[ydiff != 0].index
            # delta = pd.Series(pd.Series(ind, index=ind).diff().dt.to_pytimedelta(), index=ind).apply(lambda x: x.total_seconds())
            # ind = delta[delta > 3600].index
            # static_modes = y_tr.shift(1).loc[ind].unique()
            # transient[y.isin(static_modes)] = 0
            
            return transient
        else:
            return y

    def fit_predict(self, x=None, y=None):
        return self.predict(x, y)


class ClusterCounter():
    def predict(self, x, y=None, metric=silhouette_score, max_clusters=50, nums=200):
        x = t_discrete(t_pca(x, 0.99), nums)
        def iteration(n, x=x, y=y):
            return metric(x, MyKMeans(n).fit(x, y).predict(x))
        nn = range(2, max_clusters)
        res = Parallel(n_jobs=N_JOBS, prefer='processes')(delayed(iteration)(n) for n in nn)
        max_res = max(zip(res, nn), key=lambda x: x[0])
        print(max_res)
        print(*zip(res, nn), sep=',')
        return min(zip(res, nn), key=lambda x: abs(max_res[0] - 0.1 - x[0]) if x[1] > max_res[1] else 1e9)[1]


class MyBaseClusterizator():
    def my_score(self, x, y=None, full_output=False):
        if y is not None:
            x = pd.concat([x, y], axis=1)
        score = clusterization_score(self, x)
        self.last_score = score
        return score if full_output else score['m']

    def look_for_n_clusters(self, x, y=None, model=KMeans, max_clusters=35, n_init=10, n=10, k=3, plot=False):
        x = t_pca(x, 0.9)
        x, _ = target_split(x)
        # inertia = pd.DataFrame(index=range(1, max_clusters + 1), columns=range(n_init))
        # for n_clusters in inertia.index:
        #     for num in inertia.columns:
        #         new_model = model(n_clusters, n_init=1).fit(x)
        #         inertia.at[n_clusters, num] = np.sqrt(new_model.inertia_)
        # res = inertia
        print(f"optimal cluster count is {self.n_clusters}")
        return res


class ClusterWithoutPredict(MyBaseClusterizator):
    def predict(self, *args, **kwargs):
        return self.fit_predict(*args, **kwargs)


class ClusterAfterDiscrete():
    def predict(self, x, y):
        df1 = x.drop_duplicates()
        r = pd.Series(self.fit_predict(*target_split(df1)), index=df1.index, name=TARGET_FEATURE)
        df1_t = target_split(df1)[0].apply(lambda x: tuple(x.to_list()), axis=1)
        df_t = target_split(df)[0].apply(lambda x: tuple(x.to_list()), axis=1)
        replacer = {k: v for k, v in zip(df1_t, r)}
        res = [replacer[x] for x in df_t]
        # pd.Series(res, index = df.index, name=TARGET_FEATURE)
        return res    


class MyKMeans(MyBaseClusterizator, KMeans):
    def __init__(self, n_clusters, random_state=RANDOM_SEED, **kwargs):
        super().__init__(n_clusters=n_clusters, random_state=RANDOM_SEED, **kwargs)
        

class MyAC(ClusterAfterDiscrete, AgglomerativeClustering):
    pass


class MyDBSCAN(ClusterWithoutPredict, DBSCAN):
    pass

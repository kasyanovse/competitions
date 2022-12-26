#!/usr/bin/env python
# -*- coding: utf-8 -*-

from math import nan
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer

from utils import (DATAFILE_NAMES, TARGET_FEATURE, ORIGIN_TARGET_FUTURE,
                   TIME_FEATURE, ORIGIN_TIME_FEATURE, SOLUTION_FILENAME, load, target_split)
from features import *
from scoring import clusterization_score_by_result, clusterization_score
from models import MyKMeans, MyAC, Equalizer, TransientFinder, FeatureExtractor


DATA_FILE = Path(DATAFILE_NAMES[-1])


class Solution():
    def __init__(self, data_file=None, pipeline_funs=None, model=None, try_to_read_dump=False):
        if data_file is None:
            data_file = DATA_FILE
        if pipeline_funs is None:
            pipeline_funs = []
        self.data = load(data_file, try_to_read_dump=try_to_read_dump)
        self.pipeline_funs = pipeline_funs
        self.pipeline = Pipeline([(str(i), FunctionTransformer(fun)) for i, fun in enumerate(pipeline_funs)]
                                 + [('na_processing', FunctionTransformer(t_delna))]
                                 + ([('model', model)] if model is not None else []))
        self.model = model
        self.last_solution = None
        self.correct_solution = None

    def get_solution(self):
        self.data_processed = self.pipeline[:-1].fit_transform(self.data)
        x, y = target_split(self.data_processed)
        self.last_solution = pd.Series(self.model.fit_predict(x, y), index=x.index, name=TARGET_FEATURE)
        if y is not None:
            self.correct_solution = y.loc[x.index]
        return self.last_solution

    def score_solution(self):
        if self.last_solution is None:
            self.get_solution()
        if self.correct_solution is None:
            return None
        return clusterization_score_by_result(self.correct_solution, self.last_solution)['m']

    def solution_to_file(self, solution_filename=SOLUTION_FILENAME):
        if self.last_solution is None:
            self.get_solution()
        df = self.data.copy()
        empty_col = pd.Series(index=self.last_solution.index, name='', dtype=float)
        df = df * nan
        df = df.iloc[:, :3]
        df = pd.concat([df, empty_col, self.last_solution], axis=1)
        df[TARGET_FEATURE] = df[TARGET_FEATURE].astype(int)
        df = pd.concat([pd.DataFrame(index=[''], columns=df.columns, dtype=float), df])
        df = df.rename(columns={TARGET_FEATURE: ORIGIN_TARGET_FUTURE})
        df.index = map(lambda x: x if isinstance(x, str) else x.strftime('%d.%m.%Y %#H:%M'), df.index)
        df.index.name = ORIGIN_TIME_FEATURE
        args = '_'.join(f"{k}_{v}" for k, v in self.args.items())
        solution_filename = solution_filename.with_stem(f'_{self.__class__.__name__}'
                                                        + '_'
                                                        + args
                                                        + '_'
                                                        + datetime.now().strftime('%m_%d_%H_%M'))
        df.to_csv(solution_filename, sep=';', encoding='cp1251')
        return df


class SolutionSimple0(Solution):
    def __init__(self, n_clusters, data_file=None, **kwargs):
        self.args = {'n_clusters': n_clusters}
        super().__init__(data_file=data_file,
                         pipeline_funs=[],
                         model=MyKMeans(n_clusters),
                         **kwargs)


class SolutionSimple1(Solution):
    def __init__(self, n_clusters, data_file=None, **kwargs):
        self.args = {'n_clusters': n_clusters}
        super().__init__(data_file=data_file,
                         pipeline_funs=[t_pca099, f_diff, f_abs2, f_window_mean60_diff],
                         model=MyKMeans(n_clusters),
                         **kwargs)


class SolutionSimple2(Solution):
    def __init__(self, n_clusters, data_file=None, **kwargs):
        self.args = {'n_clusters': n_clusters}
        super().__init__(data_file=data_file,
                         pipeline_funs=[t_q_q, replace_corr, f_diff, f_window_mean20_diff],
                         model=MyKMeans(n_clusters),
                         **kwargs)


class SolutionSimple3(Solution):
    def __init__(self, n_clusters, data_file=None, **kwargs):
        self.args = {'n_clusters': n_clusters}
        super().__init__(data_file=data_file,
                         pipeline_funs=[t_to_positive, f_abs2, f_diff, f_abs2, f_window_mean30_diff, f_window_mean60_diff, f_window_mean100_diff],
                         model=MyKMeans(n_clusters),
                         **kwargs)


class SolutionSimple4(Solution):
    def __init__(self, n_clusters, data_file=None, **kwargs):
        self.args = {'n_clusters': n_clusters}
        super().__init__(data_file=data_file,
                         pipeline_funs=[t_pca099, t_min_max, f_abs2, f_diff, f_abs2, f_window_mean60_diff, f_window_mean60_diff],
                         model=MyKMeans(n_clusters),
                         **kwargs)


class SolutionSimple5(Solution):
    def __init__(self, n_clusters, data_file=None, **kwargs):
        self.args = {'n_clusters': n_clusters}
        super().__init__(data_file=data_file,
                         pipeline_funs=[t_pca099, f_abs2, f_diff2, f_window_mean30_diff, f_window_mean30_diff, t_discrete],
                         model=MyKMeans(n_clusters),
                         **kwargs)


class SolutionSimple6(Solution):
    def __init__(self, n_clusters, data_file=None, **kwargs):
        self.args = {'n_clusters': n_clusters}
        super().__init__(data_file=data_file,
                         pipeline_funs=[t_pca099, f_abs2, f_diff2, f_window_mean30_diff, f_window_mean30_diff, t_discrete],
                         model=MyAC(n_clusters),
                         **kwargs)
        
        
class SolutionMain0(Solution):
    def __init__(self, start_clusters, data_file=None, **kwargs):
        self.args = {'start_clusters': start_clusters}
        self.start_clusters = start_clusters
        super().__init__(data_file=data_file)

    def get_solution(self):
        cluster_model = MyKMeans(self.start_clusters)
        
        # preprocessing
        df0 = t_pca(self.data, 0.999)
        for fun in [f_abs2, f_diff2, f_window_mean30_diff, f_window_mean30_diff]:
            df0 = fun(df0)
        df = t_discrete(df0, 100, False)
        
        # clusterization
        df1 = df.drop_duplicates()
        r = pd.Series(cluster_model.fit_predict(*target_split(df1)), index=df1.index, name=TARGET_FEATURE)
        df1_t = target_split(df1)[0].apply(lambda x: tuple(x.to_list()), axis=1)
        df_t = target_split(df)[0].apply(lambda x: tuple(x.to_list()), axis=1)
        replacer = {k: v for k, v in zip(df1_t, r)}
        res = pd.Series([replacer[x] for x in df_t], index = df.index, name=TARGET_FEATURE)
        res = Equalizer().fit_transform(y=res)
        self.last_solution = res
        return self.last_solution


class SolutionMain1(Solution):
    def __init__(self, start_clusters, data_file=None, **kwargs):
        self.args = {'start_clusters': start_clusters}
        self.start_clusters = start_clusters
        super().__init__(data_file=data_file)

    def get_solution(self):
        cluster_model = MyKMeans(self.start_clusters)
        
        # preprocessing
        df0 = t_pca(self.data, 0.999)
        for fun in [f_abs2, f_diff2, f_window_mean30_diff, f_window_mean30_diff]:
            df0 = fun(df0)
        df = t_discrete(df0, 100, False)
        
        # clusterization
        df1 = df.drop_duplicates()
        r = pd.Series(cluster_model.fit_predict(*target_split(df1)), index=df1.index, name=TARGET_FEATURE)
        df1_t = target_split(df1)[0].apply(lambda x: tuple(x.to_list()), axis=1)
        df_t = target_split(df)[0].apply(lambda x: tuple(x.to_list()), axis=1)
        replacer = {k: v for k, v in zip(df1_t, r)}
        res = pd.Series([replacer[x] for x in df_t], index = df.index, name=TARGET_FEATURE)
        
        # equalization
        res = Equalizer().fit_transform(y=res)
        self.last_solution = res
        
        # define transient
        self.transient = TransientFinder().predict(target_split(self.data)[0], self.last_solution, use_edge=True)
        transient = self.transient
        
        x, _ = target_split(self.data.dropna())
        extractor = FeatureExtractor()
        features = extractor.extract_features(x, self.last_solution)
        self.features = features
        self.last_solution = extractor.glue_modes_by_main_features(x, self.last_solution, features)
        
        return self.last_solution


if __name__ == '__main__':
    import sys
    for i, m in enumerate((SolutionSimple0, SolutionSimple1, SolutionSimple2, SolutionSimple3, SolutionSimple4)):
        print(i)
        m(39).solution_to_file()
    # def solution_class(name):
        # return all((name.startswith('Solution'),
                    # not name.endswith('Solution')))
    # objs = {k: v for k, v in sys.modules[__name__].__dict__.items() if solution_class(k)}
    # for name, solution_class in objs.items():
        # print(name)
        # s = solution_class(15)
        # s.solution_to_file()
        # breakpoint()


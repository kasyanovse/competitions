from copy import deepcopy, copy
from datetime import datetime

import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator
from sklearn.preprocessing import FunctionTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier, StackingClassifier
from sklearn.naive_bayes import GaussianNB, MultinomialNB, ComplementNB, BernoulliNB, CategoricalNB
from catboost import CatBoostClassifier

from parameters import RANDOM_SEED, SCORERS, TARGET_FEATURE, INVERSE_TARGET_REPLACER
from preparing import MyOheHotEncoder, MyOrdinalEncoder, MyMinMaxScaler, ColumnsSorter, EmptyColFiller, MyPolynomialFeatures, one_hot_encoding
        

class CommonEstimator(BaseEstimator):
    attr_from_submodel = ('get_params', )
    
    def __init__(self, model_class, model_pars=None, transformers=None):
        self.model_class = model_class
        self.model_pars = model_pars if model_pars else dict()
        self.model = None
        self.transformers = transformers if transformers else [FunctionTransformer()]
        self._fitted = False
        self.test_res = None
    
    def __getattr__(self, attr):
        if attr in dir(self) and attr not in self.attr_from_submodel:
            return getattr(self, attr)
        elif self.model and isinstance(self.model, Pipeline):
            return getattr(self.model[-1], attr)
        elif self.model:
            return getattr(self.model, attr)
        raise AttributeError(f"unknown attribute {attr} for CommonEstimator class or model is not created yet (does not fitted)")
    
    def __sklearn_is_fitted__(self):
        return self._fitted
    
    def fit(self, x, y):
        x = copy(x)
        if self.model is None:
            self.model = self.model_class(**self.model_pars)
        self.model = Pipeline([(str(i), t) for i, t in enumerate(self.transformers)] + [('model', self.model)])
        self.model.fit(x, y)
        self._fitted = True
        return self
    
    def predict(self, x, *args):
        return self.model.predict(x)
    
    def predict_proba(self, *args):
        return self.model.predict_proba(*args)
    
    def score(self, x, y):
        return SCORERS[0](self, x, y)
    
    def set_params(self, *args, **kwargs):
        self.model.set_params(*args, **kwargs)
    
    def test(self, data_for_test):
        if not self.__sklearn_is_fitted__():
            self.fit(*data_for_test[0])
        names = ['train'] + [f"test_{i}" for i in range(1, len(data_for_test))]
        res = {name: self.score(*data_for_test[i]) for i, name in enumerate(names)}
        mi = [v for k, v in res.items() if 'test' in k]
        m = sum(mi) / len(mi)
        self.test_res = {'train': res['train'], 'test': m, 'delta': res['train'] - m, 'std': np.std(mi)}
        return pd.DataFrame([self.test_res])
    
    def predict_final(self, data_for_test, data_for_predict, refit=False):
        f = pd.concat([x[0] for x in data_for_test])
        t = pd.concat([x[1] for x in data_for_test])
        if refit:
            self.fit(f, t)
            t0 = pd.Series(self.predict(data_for_predict).squeeze(), index=data_for_predict.index)
            self.fit(pd.concat([f, data_for_predict]), pd.concat([t, t0]))
        print(f"predict: {SCORERS[0](self, f, t)}")
        t0 = (pd.DataFrame(self.predict(data_for_predict),
                           columns=['Статус'],
                           index=pd.Index(data_for_predict.index, name='ID')).iloc[:, 0]).astype(int)
        if 'country' in data_for_predict.columns:
            t0[(data_for_predict['country'] == 'китай') & (t0 == 2)] = 1
        display(pd.concat([t0.value_counts() / t0.value_counts().sum(), t.value_counts() / t.value_counts().sum()], axis=1))
        t0 = t0.replace(INVERSE_TARGET_REPLACER)
        t0.to_csv(f"result/res_{datetime.today().strftime('%Y.%m.%d_%H.%M')}.csv")
        t0.to_csv(f"res.csv")
        return t0


class CBC(CommonEstimator):
    _estimator_type = 'classifier'
    
    def __init__(self, model_pars=None, transformers=None):
        model_pars = model_pars if model_pars else dict()
        model_pars |= {'auto_class_weights': 'Balanced',
                       'verbose': 0,
                       'eval_metric': 'TotalF1:average=Macro',
                       'learning_rate': 0.01,
                       'random_seed': RANDOM_SEED,}
        model_pars |= {'colsample_bylevel': 0.06451399508826543,
                       'depth': 12,
                       'boosting_type': 'Plain',
                       'bootstrap_type': 'MVS',
                       'l2_leaf_reg': 0.30754893129585104,
                       'random_strength': 0.3192712709202957}
        super().__init__(model_class=CatBoostClassifier, model_pars=model_pars, transformers=transformers)
    
    def fit(self, x, y):
        x = copy(x)
        if isinstance(x, pd.DataFrame):
            pipe = Pipeline([(str(i), deepcopy(x)) for i, x in enumerate(self.transformers)])
            self.model_pars |= {'cat_features': list(pipe.fit_transform(x).select_dtypes(include='category').columns)}
        if len(np.unique(y)) > 2:
            self.model_pars |= {'loss_function': 'MultiClass', 'classes_count': len(np.unique(y))}
        self.model = self.model_class(**self.model_pars)
        super().fit(x, y)
        return self


class CBCt(CBC):
    _estimator_type = 'classifier'
    
    def __init__(self, model_pars=None, transformers=None):
        model_pars = model_pars if model_pars else dict()
        model_pars |= {'colsample_bylevel': 0.06451399508826543,
                       'depth': 12,
                       'boosting_type': 'Plain',
                       'bootstrap_type': 'MVS',
                       'l2_leaf_reg': 0.30754893129585104,
                       'random_strength': 0.3192712709202957}
        super().__init__(model_pars=model_pars, transformers=transformers)


class LRC(CommonEstimator):
    _estimator_type = 'classifier'
    
    def __init__(self, model_pars=None, transformers=None):
        model_pars = model_pars if model_pars else dict()
        transformers = transformers if transformers else list()
        # transformers += [FunctionTransformer(one_hot_encoding)]
        if not any([isinstance(x, MyOheHotEncoder) for x in transformers]):
            transformers += [MyOheHotEncoder()]
        model_pars |= {'C': 0.1,
                       'max_iter': 10000,
                       'class_weight': 'balanced'}
        super().__init__(model_class=LogisticRegression, model_pars=model_pars, transformers=transformers)


class ConsecutiveEstimator(CommonEstimator):
    def __init__(self, model_1, model_2):
        self.model = (model_1, model_2)
        self.replacer = [{2: 1}, {1: 0, 2: 1}]
        self.invercer = [{v: k for k, v in x.items()} for x in self.replacer]
        self._fitted = False

    def fit(self, x, y):
        x = copy(x)
        self.model[0].fit(x, y.replace(self.replacer[0]))
        self.model[1].fit(x[y != 0], y[y != 0].replace(self.replacer[1]))
        self._fitted = True
        return self

    def predict(self, x, y=None):
        res_1 = pd.Series(self.model[0].predict(x).squeeze(), index=x.index)
        temp = res_1 != 0
        res_2 = pd.Series(self.model[1].predict(x[temp]).squeeze(), index=x[temp].index)
        res_1[x[temp].index] = res_2.replace(self.invercer[1])
        return res_1.values


class ConsecutiveEstimatorProba(ConsecutiveEstimator):
    shares = (0.607259, 0.347541, 0.045200)
    
    def predict(self, x, y=None):
        res_1 = pd.Series(self.model[0].predict_proba(x)[:, 1].squeeze(), index=x.index)
        res_1 = (res_1 > res_1.quantile(self.shares[0])).astype(int)
        temp = res_1 != 0
        res_2 = pd.Series(self.model[1].predict_proba(x[temp])[:, 1].squeeze(), index=x[temp].index)
        res_2 = (res_2 > res_2.quantile(self.shares[1] / (1 - self.shares[0]))).astype(int)
        res_1[x[temp].index] = res_2.replace(self.invercer[1])
        return res_1.values


class StackEstimator(CommonEstimator):
    def __init__(self, estimators, final_estimator=None, **kwargs):
        model_pars = {'estimators': [(str(i), x) for i, x in enumerate(estimators)],
                      'final_estimator': final_estimator,
                      'stack_method': 'predict_proba',
                      'cv': 'prefit' if all([x.__sklearn_is_fitted__() for x in estimators]) else 2} | kwargs
        super().__init__(model_class=StackingClassifier, model_pars=model_pars)
    


# class ConsecutiveFunEstimator():
#     def __init__(self, model_fun):
#         self.fun = lambda *args: model_fun(*args, **kwargs)
#         self.replacer = [{2: 1}, {1: 0, 2: 1}]
#         self.invercer = [{v: k for k, v in x.items()} for x in self.replacer]

#     def fit(self, x, y):
#         res_1 = self.fun([(x, y.replace(self.replacer[0]))])
#         res_2 = self.fun([(x[y != 0], y[y != 0].replace(self.replacer[1]))])
#         self.model = (res_1['model'], res_2['model'])
#         return self

#     def predict(self, x, y=None):
#         res_1 = pd.Series(self.model[0].predict(x).squeeze(), index=x.index)
#         temp = res_1 != 0
#         res_2 = pd.Series(self.model[1].predict(x[temp]).squeeze(), index=x[temp].index)
#         res_1[x[temp].index] = res_2.replace(self.invercer[1])
#         return res_1.values


# def consecutive_prediction(model_fun, data_for_eval, **kwargs):
#     return eval_pipe(ConsecutiveFunEstimator(model_fun), data_for_eval)


# def bagging_prediction(model_fun, data_for_eval, **kwargs):
#     _model_pars = {'n_estimators': 10,
#                    'random_state': RANDOM_SEED,
#                    'verbose': 0}
#     pipe = model_fun(**kwargs)
#     model = BaggingClassifier(pipe[-1], **_model_pars)
#     return eval_pipe(model, data_for_eval)
    

# def prepare_n_run_model(data_for_eval=None, model=None, model_pars=None, transformers=None,
#                         _model=None, _transformers=None, _model_pars=None, **kwargs):
#     if not transformers:
#         transformers = _transformers
#     if model is None:
#         if model_pars is None:
#             model_pars = _model_pars
#         else:
#             model_pars = _model_pars | model_pars
#         model = _model(**model_pars)
#     pipe = Pipeline([(str(i), t) for i, t in enumerate(transformers)] + [('model', model)])
#     return eval_pipe(pipe, data_for_eval, **kwargs)


# def simple_bayes_g(data_for_eval=None, **kwargs):
#     return prepare_n_run_model(data_for_eval, **kwargs, _model=GaussianNB,
#                                _transformers={},  _model_pars={})

# def simple_bayes_m(data_for_eval=None, **kwargs):
#     return prepare_n_run_model(data_for_eval, **kwargs, _model=MultinomialNB,
#                                _transformers={},  _model_pars={})

# def simple_bayes_c(data_for_eval=None, **kwargs):
#     return prepare_n_run_model(data_for_eval, **kwargs, _model=ComplementNB,
#                                _transformers={},  _model_pars={})

# def simple_bayes_b(data_for_eval=None, **kwargs):
#     return prepare_n_run_model(data_for_eval, **kwargs, _model=BernoulliNB,
#                                _transformers={},  _model_pars={})

# def simple_bayes_cat(data_for_eval=None, **kwargs):
#     return prepare_n_run_model(data_for_eval, **kwargs, _model=CategoricalNB,
#                                _transformers={},  _model_pars={})


# def ensemble_predict(result, data):
#     r = pd.concat([pd.Series(row['model'].predict(data[list(row['cols'])]).squeeze()) for _, row in result.iterrows()], axis=1)
#     t0 = r.mode(axis=1)[0]
#     t0.name = 'Статус'
#     t0.index = data.index
#     t0.index.name = 'ID'
#     return r, t0


# def simple_linear(data_for_eval=None, **kwargs):
#     _model=LogisticRegression
#     _transformers = (kwargs['_transformers'] if '_transformers' in kwargs else []) + [MyMinMaxScaler(), FunctionTransformer(one_hot_encoding)]
#     _model_pars = (kwargs['_model_pars'] if '_model_pars' in kwargs else {}) | {'C': 0.01,
#                  'max_iter': 1000,
#                  'class_weight': 'balanced'}
#     for k in [x for x in ('classes_count', 'cat_features', '_transformers', '_model_pars') if x in kwargs]:
#         del kwargs[k]
#     return prepare_n_run_model(data_for_eval, **kwargs, _model=_model, _transformers=_transformers,  _model_pars=_model_pars)


# def simple_forest(data_for_eval=None, **kwargs):
#     _model=RandomForestClassifier
#     _transformers = (kwargs['_transformers'] if '_transformers' in kwargs else []) + [MyMinMaxScaler()]
#     _model_pars = (kwargs['_model_pars'] if '_model_pars' in kwargs else {}) | {'max_depth': 20,
#                  'n_estimators': 100,
#                  'random_state': RANDOM_SEED,
#                  'class_weight': 'balanced'}
#     for k in [x for x in ('classes_count', 'cat_features', '_transformers', '_model_pars') if x in kwargs]:
#         del kwargs[k]
#     return prepare_n_run_model(data_for_eval, **kwargs, _model=_model, _transformers=_transformers,  _model_pars=_model_pars)


# def catboost(data_for_eval=None, **kwargs):
#     _model = CatBoostClassifier
#     _transformers = (kwargs['_transformers'] if '_transformers' in kwargs else [])# + [MyMinMaxScaler()]
#     if 'cat_features' not in kwargs:
#         if '_transformers' in kwargs:
#             data = Pipeline([(str(i), i) for i in deepcopy(kwargs['_transformers'])]).fit_transform(data_for_eval[-1][0])
#         else:
#             data = data_for_eval[-1][0]
#         cat_features = list(data.select_dtypes(include='category').columns)
#     else:
#         cat_features = kwargs['cat_features']
#     _model_pars = (kwargs['_model_pars'] if '_model_pars' in kwargs else {}) | {'verbose': 0,
#                      'classes_count': kwargs['classes_count'] if 'classes_count' in kwargs else len(data_for_eval[0][1].unique()),
#                      'loss_function': 'MultiClass',
#                      'auto_class_weights': 'Balanced',
#                      'eval_metric': 'TotalF1:average=Macro',
#                      'learning_rate': 0.01,
#                      'random_seed': RANDOM_SEED,
#                      'cat_features': cat_features}
#     for k in [x for x in ('classes_count', 'cat_features', '_transformers', '_model_pars') if x in kwargs]:
#         del kwargs[k]
#     return prepare_n_run_model(data_for_eval, **kwargs, _model=_model, _transformers=_transformers,  _model_pars=_model_pars)

# def simple_catboost(data_for_eval=None, **kwargs):
#     _model_pars={'depth': 8,
#                  'iterations': 800,
#                  'leaf_estimation_method': 'Newton',
#                  'boosting_type': 'Ordered',}
#     return catboost(data_for_eval, _model_pars=_model_pars, **kwargs)


# def gpu_catboost(data_for_eval=None, **kwargs):
#     _model_pars={'depth': 7,
#                  'task_type': 'GPU',
#                  'devices': '0',}
#     return catboost(data_for_eval, _model_pars=_model_pars, **kwargs)


# def fast_catboost(data_for_eval=None, **kwargs):
#     _model_pars={'depth': 7}
#     return catboost(data_for_eval, _model_pars=_model_pars, **kwargs)


# def very_fast_catboost(data_for_eval=None, **kwargs):
#     _model_pars={'depth': 4,
#                  'learning_rate': 0.5,}
#     return catboost(data_for_eval, _model_pars=_model_pars, **kwargs)


# def simple_stacking_catboost(estimators, data_fro_eval=None, **kwargs):
#     model = CatBoostClassifier(learning_rate=0.01,
#                                verbose=0,
#                                classes_count=kwargs['classes_count'] if 'classes_count' in kwargs else len(data_for_eval[0][1].unique()),
#                                loss_function='MultiClass',
#                                auto_class_weights='Balanced')
#     return eval_pipe(StackingClassifier([(str(i), x) for i, x in enumerate(estimators)], model), data_for_eval)


# def predict_on_test(model, ft=None, fp=None, refit=True):
#     if ft is None or fp is None:
#         f, ft, fp = get_full_prepared_data_with_upsample()
#     f = pd.concat([x[0] for x in ft])
#     t = pd.concat([x[1] for x in ft])
#     if refit:
#         print('start fitting')
#         model.fit(f, t)
#         t0 = pd.Series(model.predict(fp).squeeze(), index=fp.index)
#         print('continue fitting')
#         model.fit(pd.concat([f, fp]), pd.concat([t, t0]))
#     print(f"predict: {SCORERS[0](model, f, t)}")
#     t0 = (pd.DataFrame(model.predict(fp), columns=['Статус'], index=pd.Index(fp.index, name='ID')).iloc[:, 0]).astype(int)
#     if 'country' in fp.columns:
#         t0[(fp['country'] == 'китай') & (t0 == 2)] = 1
#     display(pd.concat([t0.value_counts() / t0.value_counts().sum(), t.value_counts() / t.value_counts().sum()], axis=1))
#     t0 = t0.replace(INVERSE_TARGET_REPLACER)
#     t0.to_csv(f"result/res_{datetime.today().strftime('%Y.%m.%d_%H.%M')}.csv")
#     t0.to_csv(f"res.csv")
#     return t0


# def eval_on_train_test(pipe, data):
#     e = lambda x: SCORERS[0](pipe, *x)
#     names = ['train'] + [f"test_{i}" for i in range(1, len(data))]
#     res = {name: e(data[i]) for i, name in enumerate(names)}
#     mi = [v for k, v in res.items() if 'test' in k]
#     m = sum(mi) / len(mi)
#     return {'train': res['train'], 'test': m, 'delta': res['train'] - m, 'std': np.std(mi)}


# def eval_pipe(pipe, data_for_eval=None, **kwargs):
#     if data_for_eval is None or len(data_for_eval) == 0 or len(data_for_eval[0][0]) == 0:
#         return pipe
#     if not hasattr(pipe, 'fitted') or not pipe.fitted:
#         pipe.fit(*data_for_eval[0])
#     if len(data_for_eval) == 1:
#         return pipe
#     return eval_on_train_test(pipe, data_for_eval) | {'model': pipe}
from math import nan

import numpy as np
import pandas as pd

from sklearn.base import TransformerMixin
from sklearn.preprocessing import PolynomialFeatures, OrdinalEncoder, OneHotEncoder
from sklearn.preprocessing import FunctionTransformer

from parameters import TARGET_FEATURE


class ColumnsSorter(TransformerMixin):
    def transform(self, x, y=None):
        return x[sorted(x.columns)]


class EmptyColFiller(TransformerMixin):
    def transform(self, x, y=None):
        for col in x.columns:
            if np.isinf(x[col]).any():
                x[col][np.isinf(x[col])] = nan
            if x[col].isna().all():
                x[col] = x[col].fillna(0)
        return x


class Scaler(TransformerMixin):
    def __init__(self, scaler):
        self.scaler = scaler()
    
    def fit(self, x, y=None):
        x = x.select_dtypes(exclude='category')
        self.cols = x.columns
        self.scaler.fit(x)
        return self
    
    def transform(self, x, y=None):
        x_cat = x = x.select_dtypes(include='category')
        x = x.select_dtypes(exclude='category')
        if x.columns != self.cols:
            raise ValueError('scaler was fitted on other features')
        return pd.concat([pd.DataFrame(self.scaler.transform(x), index=x.index, columns=x.columns), x_cat], axis=1)


class MyMinMaxScaler(TransformerMixin):
    def fit(self, df, *args):
        self.coeffs = dict()
        ncat_cols = [x for x in df.select_dtypes(exclude='category').columns if x != TARGET_FEATURE]
        for col in ncat_cols:
            self.coeffs[col] = (df[col].quantile(0.25), df[col].quantile(0.75))
            if abs(self.coeffs[col][1] - self.coeffs[col][0]) < 1e-2:
                self.coeffs[col] = (df[col].quantile(0.02), df[col].quantile(0.98))
            #self.coeffs[col] = (df[col].min(), df[col].max())
        return self
    
    def transform(self, df, *args):
        for col, (a, b) in self.coeffs.items():
            #df[col] = (df[col] - a) / (b - a)
            df[col] = (df[col] - a) / (b - a) * 2 - 1
        return df


def one_hot_encoding(df):
    df1, df2 = df.select_dtypes(include='category'), df.select_dtypes(exclude='category')
    df1 = pd.get_dummies(df1, drop_first=True)
    for col in df1.columns:
        df1[col] = df1[col].astype('category')
    return pd.concat([df2, df1], axis=1)


class MyOheHotEncoder(TransformerMixin):
    def fit(self, df, y=None):
        if isinstance(df, pd.DataFrame):
            df = df.copy()
            df_cat, df_ncat = df.select_dtypes(include='category'), df.select_dtypes(exclude='category')
            df_cat = pd.get_dummies(df_cat, drop_first=True)
            self.new_cols = df_cat.columns
        return self
    
    def transform(self, df, y=None):
        if isinstance(df, pd.DataFrame):
            df = df.copy()
            df_cat, df_ncat = df.select_dtypes(include='category'), df.select_dtypes(exclude='category')
            df_cat = pd.get_dummies(df_cat, drop_first=True)
            for col in self.new_cols:
                if col not in df_cat.columns:
                    df_cat[col] = 0
            for col in df_cat.columns:
                if col not in self.new_cols:
                    df_cat = df_cat.drop(columns=[col])
            return pd.concat([df_ncat, df_cat], axis=1)
        else:
            return df


def ordinal_encoding(df):
    df1, df2 = df.select_dtypes(include='category'), df.select_dtypes(exclude='category')
    e = OrdinalEncoder()
    e.fit(df1)
    c = {col: {x: j for j, x in enumerate(e.categories_[i])} | {j: x for j, x in enumerate(e.categories_[i])} for i, col in enumerate(df1.columns)}
    df1 = pd.DataFrame(e.transform(df1).astype(int), index=df1.index, columns=df1.columns)
    for col in df1.columns:
        df1[col] = df1[col].astype('category')
    return pd.concat([df2, df1], axis=1), c


class MyOrdinalEncoder(TransformerMixin):
    def fit(self, x, y=None):
        x1 = x.select_dtypes(include='category')
        self.cat = {col: {y: x for x, y in enumerate(x1[col].cat.categories)} for col in x1.columns}
        return self
    
    def transform(self, x, y=None):
        for col in self.cat:
            x[col] = x[col].astype('object')
            x[col] = x[col].replace(self.cat[col])
            x[col] = x[col].astype('category')
        return x

    
class MyPolynomialFeatures(TransformerMixin):
    def transform(self, x, y=None):
        df = x.drop(columns=[TARGET_FEATURE]) if TARGET_FEATURE in x.columns else x.copy()
        f_num = df.select_dtypes(include=[np.number])
        f_not_num = df.select_dtypes(exclude=[np.number])
        pf = PolynomialFeatures(degree=2, include_bias=False)
        f_nums = pf.fit_transform(f_num)
        f_nums = pd.DataFrame(f_nums, index=f_num.index,
                              columns=pf.get_feature_names(f_num.columns))
        return pd.concat([f_nums, f_not_num] + (x[TARGET_FEATURE] if TARGET_FEATURE in x.columns else []), axis=1)


def col_cutter(cols, transformer=True):
    def temp(x, *args, cols=cols):
        return x.drop(columns=[y for y in x.columns if y in cols])
    return FunctionTransformer(temp) if transformer else temp


def col_retainer(cols, transformer=True):
    def temp(x, *args, cols=cols):
        return x.drop(columns=[y for y in x.columns if y not in cols])
    return FunctionTransformer(temp) if transformer else temp
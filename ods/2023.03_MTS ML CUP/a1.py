""" tools for 1 approach """

import numpy as np
import pandas as pd
from tqdm import tqdm

from settings import *
from load import load_target
from a1_features import FUNS, get_common_feature, get_common_feature_dict 


def get_data(count_of_rows=None, funs=FUNS):
    if count_of_rows is not None:
        raise NotImplementedError('i not sure that it is working properly for now because there are a lot of changes at last time')
    user_ids = (load_target()['user_id'] if count_of_rows is None
                                         else load_target()['user_id'].iloc[:count_of_rows])
    dfis = get_common_feature_dict()
    return pd.concat([fun(dfis, allow_additional_calculations=(count_of_rows is not None)) for fun in funs.values()], axis=1, join='inner')


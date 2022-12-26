# -*- coding: utf-8 -*-

from pathlib import Path

import numpy as np

PATH = dict()
PATH['main'] = Path(__file__).parent


PATH['dump'] = PATH['main'] / 'dump'
if not PATH['dump'].exists():
    PATH['dump'].mkdir()
    

PATH['data'] = dict()
# dict with new data keys and path to data for it
data_pathes = dict()
data_pathes['origin'] = PATH['main'] / 'data'
# first approach, try to work with data amplitudes as is
# data_pathes['amp_1'] = PATH['main'] / 'data'
# second approach, try to split data amplitudes to small chunks
data_pathes['amp_2'] = PATH['main'] / 'data'
for data_key in data_pathes:
    PATH['data'][data_key] = dict()
    PATH['data'][data_key]['main'] = data_pathes[data_key] / data_key
    PATH['data'][data_key]['train'] = PATH['data'][data_key]['main'] / 'train'
    PATH['data'][data_key]['test'] = PATH['data'][data_key]['main'] / 'test'
    PATH['data'][data_key]['generated'] = PATH['data'][data_key]['main'] / 'generated'

    for data_path in PATH['data'][data_key].values():
        if not data_path.exists():
            data_path.mkdir()

SETTINGS = dict()

SETTINGS['random_seed'] = 0
SETTINGS['random_generator'] = np.random.RandomState(SETTINGS['random_seed'])

SETTINGS['labels_file_name'] = '_labels.csv'
SETTINGS['snr_file_name'] = '_snr.csv'
SETTINGS['labels_id'] = 'id'
SETTINGS['labels_label'] = 'target'
SETTINGS['target_to_skip'] = -1

SETTINGS['image_shape'] = (140, 4000)

SETTINGS['base_format'] = '.npy'

SETTINGS['fig_size'] = 12, 6
SETTINGS['wide_fig_size'] = 12, 3


# amp_1
SETTINGS['image_shape_for_amp_1'] = SETTINGS['image_shape']
SETTINGS['base_format_for_amp_1'] = SETTINGS['base_format']


# amp_2
# SETTINGS['base_format_for_amp_2'] = '.png'
SETTINGS['base_format_for_amp_2'] = '.npy'
SETTINGS['image_shape_for_amp_2'] = (180, 360)
# SETTINGS['max_val_for_data_amp_2'] = 65534
SETTINGS['max_val_for_data_amp_2'] = int(2 ** 30)


# catboost approach
PATH['catboost_db_file'] = PATH['dump'] / 'catboost_base.pickle'
PATH['catboost_tune_pars'] = PATH['dump'] / 'catboost_tune_pars.pickle'

SETTINGS['catboost'] = dict()
SETTINGS['catboost']['target_feature'] = SETTINGS['labels_label']
SETTINGS['edge_amp_2'] = 10
# -*- coding: utf-8 -*-

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from settings import PATH

def create_data_generators(validation_sample=True):
    datagen_pars0 = {'width_shift_range': 0.2, 'height_shift_range': 0.2,
                     'shear_range': 0.2, 'zoom_range': 0.2, 'fill_mode': 'reflect',
                     'horizontal_flip': True, 'vertical_flip': True}
    datagen_pars = {'target_size': SETTINGS['image_shape_for_amp_2'], 'batch_size': 16*4,
                    'class_mode': 'sparse', 'seed': SETTINGS['random_seed'],
                    'color_mode': 'grayscale'}
    
    dg = ImageDataGenerator(validation_split=0.2, **datagen_pars0)
    path = PATH['data']['amp_2']['generated']
    df = pd.read_csv(path / SETTINGS['labels_file_name'], dtype=str)
    df[SETTINGS['labels_id']] = df[SETTINGS['labels_id']].apply(lambda x: x + SETTINGS['base_format_for_amp_2'])
    
    if validation_sample:
        train_datagen_flow = dg.flow_from_dataframe(df, x_col=SETTINGS['labels_id'], y_col=SETTINGS['labels_label'],
                                                    directory=path, subset='training', **datagen_pars)
        val_datagen_flow = dg.flow_from_dataframe(df, x_col=SETTINGS['labels_id'], y_col=SETTINGS['labels_label'],
                                                  directory=path, subset='validation', **datagen_pars)
        return train_datagen_flow, val_datagen_flow
    else:
        return dg.flow_from_dataframe(df, x_col=SETTINGS['labels_id'], y_col=SETTINGS['labels_label'],
                                      directory=path, **datagen_pars)


# if __name__ == '__main__':
#     pass
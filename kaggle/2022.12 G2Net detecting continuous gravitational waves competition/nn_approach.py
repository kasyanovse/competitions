import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.random import set_seed
from tensorflow.nn import moments, conv2d
from tensorflow.keras import Model, Sequential
from tensorflow.keras.metrics import AUC
from tensorflow.keras.losses import BinaryCrossentropy, CategoricalCrossentropy, SparseCategoricalCrossentropy, Poisson
from tensorflow.keras.optimizers import Adam, SGD, RMSprop
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import image_dataset_from_directory
from tensorflow.keras.layers import ReLU, BatchNormalization, Conv2D, Add, Input, AveragePooling2D, MaxPooling2D, Flatten, Dense, ReLU, ELU

from settings import PATH, SETTINGS
from data_convert import retarget_data_approach_2

EDGE = SETTINGS['edge_amp_2']

class MyNN():
    def __init__(self, edge=EDGE):
        self.edge=edge
        self.model = get_nn_model()
        
    def fit(self, epochs=40, epoch_steps=10, val_steps=5):
        history = self.model.fit(self.train_datagen_flow,
                                 validation_data=self.val_datagen_flow,
                                 steps_per_epoch=epoch_steps,
                                 validation_steps=val_steps,
                                 epochs=epochs)
        self.history = history
        return
    
    def read_data(self, edge=0):
        datagen_pars0 = {'width_shift_range': 0.2, 'height_shift_range': 0.2,
                         'shear_range': 0.2, 'zoom_range': 0.2, 'fill_mode': 'reflect',
                         'horizontal_flip': True, 'vertical_flip': True}
        datagen_pars = {'target_size': SETTINGS['image_shape_for_amp_2'], 'batch_size': 16*4,
                        'class_mode': 'sparse', 'seed': SETTINGS['random_seed'],
                        'color_mode': 'grayscale'}
        dg = ImageDataGenerator(validation_split=0.2, **datagen_pars0)
        path = PATH['data']['amp_2']['generated']
        # df = pd.read_csv(path / SETTINGS['labels_file_name'], dtype=str)
        df = pd.read_csv(path / SETTINGS['snr_file_name'])
        df[SETTINGS['labels_label']] = (df[SETTINGS['labels_label']] > self.edge).astype(int)
        df[SETTINGS['labels_id']] = df[SETTINGS['labels_id']].apply(lambda x: x + SETTINGS['base_format_for_amp_2'])
        train_datagen_flow = dg.flow_from_dataframe(df, x_col=SETTINGS['labels_id'], y_col=SETTINGS['labels_label'],
                                                    directory=path, subset='training', **datagen_pars)
        val_datagen_flow = dg.flow_from_dataframe(df, x_col=SETTINGS['labels_id'], y_col=SETTINGS['labels_label'],
                                                  directory=path, subset='validation', **datagen_pars)
        self.train_datagen = train_datagen_flow
        self.val_datagen = val_datagen_flow


def get_nn_model():
    def relu_bn(inputs):
        # relu = ReLU()(inputs)
        relu = ELU()(inputs)
        # return BatchNormalization()(relu)
        return relu

    def residual_block(x, strides, filters, kernel_size=3):
        y = Conv2D(kernel_size=kernel_size,
                   strides=strides,
                   filters=filters,
                   padding="same")(x)
        y = relu_bn(y)
        y = Conv2D(kernel_size=kernel_size,
                   strides=1,
                   filters=filters,
                   padding="same")(y)

        if strides > 1:
            x = Conv2D(kernel_size=2,
                       strides=strides,
                       filters=filters,
                       padding="same")(x)
        out = Add()([x, y])
        out = relu_bn(out)
        return out 

    shape = SETTINGS['image_shape_for_amp_2'] + (1, )
    inputs = Input(shape=shape)

    num_blocks_list = [1] * 5
    first_conv_kernel = 7
    num_filters = 16

    # num_blocks_list = [3] * 3
    # first_conv_kernel = 7
    # num_filters = 32

    # num_blocks_list = [2] * 4
    # first_conv_kernel = 7
    # num_filters = 32

    # Stemming
    t = BatchNormalization()(inputs)
    t = Conv2D(kernel_size=first_conv_kernel,
               strides=1,
               filters=num_filters,
               padding="same")(t)
    t = relu_bn(t)

    # Blocks
    for i in range(len(num_blocks_list)):
        num_blocks = num_blocks_list[i]
        if shape[0] % 2 == 0:
            strides = 2
        elif shape[0] % 3 == 0:
            strides = 3
        elif shape[0] % 5 == 0:
            strides = 5
        else:
            strides = 1
        for j in range(num_blocks):
            real_strides = strides if (j==0 and i!=0) else 1
            t = residual_block(t, strides=real_strides, filters=num_filters)
        shape = [shape[0] / real_strides, shape[1] / real_strides, shape[2]]
        num_filters *= 2

    # Ending
    # if strides < 5:
    #     t = MaxPooling2D(strides)(t)
    t = Flatten()(t)
    # t = Dense(10, activation='elu')(t)
    # t = Dense(10, activation='elu')(t)
    t = Dense(10, activation='elu')(t)
    outputs = Dense(1, activation='sigmoid')(t)

    model = Model(inputs, outputs)
    model.compile(loss=BinaryCrossentropy(), optimizer=Adam(learning_rate=1e-5), metrics=AUC())
    return model
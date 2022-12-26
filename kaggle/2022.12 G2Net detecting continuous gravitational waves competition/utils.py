# -*- coding: utf-8 -*-
""" some usefull functions """

import pickle

import h5py
import numpy as np


def load_data_from_hdf5(file_name):
    with h5py.File(file_name, 'r') as hf:
        key = list(hf.keys())[0]
        datas = [hf[key][subkey]['SFTs'] for subkey in ('H1', 'L1')]
        shape = (min([x.shape[0] for x in datas]), min([x.shape[1] for x in datas]))
        freq = hf[key]['frequency_Hz']
        time = hf[key]['H1']['timestamps_GPS']
        data = {'id': key,
                'freq_data_1': freq[0],
                'freq_data_2': freq[-1],
                'freq_data_3': freq[1] - freq[0],
                'time_data_1': time[0],
                'time_data_2': time[-1],
                'time_data_3': time[1] - time[0],
                'data': np.concatenate([np.reshape(x[:shape[0], :shape[1]], (shape[0], shape[1], 1)) for x in datas], 2)}
    return data


def write(file_name, data):
    with open(file_name, 'wb') as handle:
        pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)


def read(file_name):
    with open(file_name, 'rb') as handle:
        return pickle.load(handle)

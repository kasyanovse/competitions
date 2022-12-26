# -*- coding: utf-8 -*-
""" funs for convert origin and generated data to common format """

from pathlib import Path
import shutil
from itertools import chain
from math import ceil

import numpy as np
import pandas as pd
from tqdm import tqdm
from PIL import Image

from settings import PATH, SETTINGS
from utils import load_data_from_hdf5

MAX_VAL = SETTINGS['max_val_for_data_amp_2']


def convert_db_with_generated_data_info_to_labels(path=None):
    if path is None:
        path = PATH['data']['origin']['generated']
    t = pd.read_csv(path / 'db.csv', index_col=0)[['id', 'target']].set_index('id')['target']
    df = pd.DataFrame([(int(x.stem), int(t.loc[int(x.stem)])) for x in path.glob('*.npy')], columns=['id', 'target']).set_index('id').sort_index()
    df.to_csv(path / SETTINGS['labels_file_name'])


def copy_labels_file(key, data_key):
    new_file = PATH['data'][data_key][key] / SETTINGS['labels_file_name']
    if (PATH['data']['origin'][key] / SETTINGS['labels_file_name']).exists():
        shutil.copy(PATH['data']['origin'][key] / SETTINGS['labels_file_name'], new_file)
        return new_file


def save_to_base_format(file_name, destination_path, data, base_format):
    new_file_name = destination_path / f"{file_name.stem}{base_format}"
    if not new_file_name.exists():
        if base_format == '.npy':
            np.save(new_file_name, data)
        elif base_format == '.png':
            if not isinstance(data, Image):
                data = Image.fromarray(data)
            data.save(new_file_name)
        else:
            raise NotImplementedError()


def prepare_data_common(process_data, data_key, base_format, forbidden_stems=[]):
    def temp1(file_name):
        data = load_data_from_hdf5(file_name)
        return process_data(data['data'])
    
    def temp2(file_name):
        data = np.load(file_name)
        return process_data(data)
    
    for key, fun, mask in zip(('train',  'test',  'generated'),
                              (temp1,    temp1,    temp2),
                              ('*.hdf5', '*.hdf5', '*.npy')):
        print(' ' * 4, key)
        for file_name in tqdm(PATH['data']['origin'][key].glob(mask)):
            stem = file_name.stem
            if (file_name.stem in forbidden_stems
                or file_name.with_stem(stem + '_1') in forbidden_stems
                or file_name.with_stem(stem + '_2') in forbidden_stems):
                continue
            data = fun(file_name)
            data1, data2 = data[:, :, 0], data[:, : ,1]
            save_to_base_format(file_name.with_stem(stem + '_1'), PATH['data'][data_key][key], data1, base_format)
            save_to_base_format(file_name.with_stem(stem + '_2'), PATH['data'][data_key][key], data2, base_format)
        label_file = copy_labels_file(key, data_key)
        if label_file:
            df = pd.read_csv(label_file)
            dfs = [df.copy(), df.copy()]
            dfs[0][SETTINGS['labels_id']] =  df[SETTINGS['labels_id']].apply(lambda x: str(x) + '_1')
            dfs[1][SETTINGS['labels_id']] =  df[SETTINGS['labels_id']].apply(lambda x: str(x) + '_2')
            df = pd.concat(dfs).set_index(SETTINGS['labels_id']).sort_index()
            df.to_csv(label_file)


def sort_data_to_folders_by_target(path, suffix):
    new_folders = []
    path = Path(path)
    if not (path / SETTINGS['labels_file_name']).exists():
        print(' ' * 8, f"file {path / SETTINGS['labels_file_name']} .csv with targets does not exists")
        print(' ' * 8, 'sorting is not conducted')
        return
    target = pd.read_csv(path / SETTINGS['labels_file_name'])
    targeted_files = {t: target[target[SETTINGS['labels_label']] == t][SETTINGS['labels_id']].values for t in target[SETTINGS['labels_label']].unique()}
    for t, files in targeted_files.items():
        print(' ' * 8, 'new target folder', t)
        if SETTINGS['target_to_skip'] and t == SETTINGS['target_to_skip']:
            print('target', SETTINGS['target_to_skip'], 'is skipped')
            for file in tqdm(files):
                file_path = (path / (file + suffix))
                if file_path.exists():
                    file_path.unlink()
            print('all skipped data is deleted')
            continue
        subpath = (path / str(t))
        new_folders.append(subpath)
        if not subpath.exists():
            subpath.mkdir()
        for file in tqdm(files):
            file_name = (str(file) + suffix)
            file_path = path / file_name
            if not file_path.exists():
                # print('file', file_name, 'does not exists')
                continue
            new_file_path = subpath / file_name
            if not new_file_path.exists():
                shutil.move(file_path, new_file_path)
    return new_folders
                
                
def convert_npy_to_png(path, delete_npy=False):
    for file in path.glob('*.npy'):
        array = np.load(file)
        image = Image.fromarray(array)
        image.save(file.with_suffix('.png'))
        if delete_npy:
            file.unlink()


def approach_1():
    approach_name = 'amp_1'
    base_format = '.npy'
    
    convert_db_with_generated_data_info_to_labels()
    
    def process_data(data):
        d = np.abs(data)
        mm = (d.min(), d.max())
        d = (d - mm[0]) / (mm[1] - mm[0]) * MAX_VAL
        d = d.astype(np.uint16)
        return d
    
    forbidden_stems = []
    for key in ('train', 'generated'):
        files = PATH['data'][approach_name][key].rglob('*' + base_format)
        forbidden_stems.extend([x.stem for x in files])
    
    print('data converting')
    prepare_data_common(process_data, approach_name, base_format, forbidden_stems=forbidden_stems)
    
    print('data sorting')
    for key in ('train', 'generated'):
        print(' ' * 4, key)
        new_folders = sort_data_to_folders_by_target(PATH['data'][approach_name][key], base_format)
        for new_folder in new_folders:
            print(' ' * 4, 'convert npy to image in', new_folder)
            convert_npy_to_png(new_folder)


def process_data_approach_2(data, max_val=MAX_VAL, to_uint=False):
    d = np.abs(data)
    mm = (d.min(), d.max())
    d = (d - mm[0]) / (mm[1] - mm[0]) * max_val
    # if to_uint:
    #     d = d.astype(np.uint16)
    # else:
    #     d = d.astype(int)
    d = d.astype(int)
    return d
            
            
def approach_2(snr_edge=0):
    approach_name = 'amp_2'
    base_format = '.png'
    path_to_origin = PATH['data']['origin']
    path_to_data = PATH['data'][approach_name]
    
    print('generated data')
    new_labels = []
    for subfolder in (path_to_origin['generated']).glob('*'):
        print(' ' * 4, 'processing', subfolder)
        subname = subfolder.name
        db = pd.read_csv(subfolder / 'db.csv').set_index('id')
        for (name, target), (_, snr) in tqdm(zip(db['target'].iteritems(), db['snr'].iteritems())):
            name = str(int(name))
            #target = int(target)
            target = snr > snr_edge
            file = subfolder / f"{name}.npy"
            if file.exists():
                array = np.load(file)
                for num_data, data in enumerate((array[:,:,0], array[:,:,1])):
                    data = np.concatenate(process_data_approach_2(data[:, :, 0], to_uint=True),
                                          process_data_approach_2(data[:, :, 1], to_uint=True), axis=2)
                    save_path = (PATH['data'][approach_name]['generated']
                                 / (subname + '_' + name + f"_{num_data}" + SETTINGS[f'base_format_for_amp_2']))
                    if not save_path.parent.exists():
                        save_path.parent.mkdir(parents=True)
                    new_labels.append({SETTINGS['labels_id']: save_path.stem, SETTINGS['labels_label']: int(target)})
                    # if save_path.exists():
                    #     continue
                    if SETTINGS['base_format_for_amp_2'] == '.png':
                        Image.fromarray(data).save(save_path)
                    elif SETTINGS['base_format_for_amp_2'] == '.npy':
                        np.save(save_path, data)
                    else:
                        raise NotImplementedError()
    pd.DataFrame(new_labels).set_index(SETTINGS['labels_id']).to_csv(PATH['data'][approach_name]['generated'] / SETTINGS['labels_file_name'])
    
    db = read_snr_from_origin()
    db.to_csv(path_to_data['generated'] / SETTINGS['snr_file_name'])
    

def read_snr_from_origin():
    dbs = []
    for folder in PATH['data']['origin']['generated'].glob('*'):
        name = folder.stem
        db = pd.read_csv(folder / 'db.csv')
        db = db[['id', 'snr']]
        db['id_0'] = db['id'].apply(lambda x: f"{name}_{x:n}_0")
        db['id_1'] = db['id'].apply(lambda x: f"{name}_{x:n}_1")
        db = pd.concat([db[['id_0', 'snr']].rename(columns={'id_0': 'id'}),
                        db[['id_1', 'snr']].rename(columns={'id_1': 'id'})])
        db = db.set_index('id').sort_index()
        dbs.append(db)
    db = pd.concat(dbs)
    db = db.rename(columns={'snr': SETTINGS['labels_label']})
    return db

def retarget_data_approach_2(edge):
    db = read_snr_from_origin
    db = (db > edge).astype(int)
    db.to_csv(PATH['data']['amp_2']['generated'] / SETTINGS['labels_file_name'])

if __name__ == '__main__':
    approach_2()

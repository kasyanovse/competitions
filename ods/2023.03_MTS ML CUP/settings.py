from pathlib import Path
import multiprocessing


RANDOM_SEED = 0


def prepare_folder(folder):
    if not folder.exists():
        folder.mkdir()
    return folder

APPROACH_COUNT = 5


path = dict()
path['main'] = Path(__file__).resolve().parent

# pathes to data
path['data'] = dict()
path['data']['main'] = prepare_folder(path['main'] / 'data')
# nonprocessed data from orgs
path['data']['origin_data'] = prepare_folder(path['data']['main'] / 'origin_data')
path['data']['origin_data_file_target'] = path['data']['origin_data'] / 'public_train.pqt'
# data in a convenient form
path['data']['processed_data'] = prepare_folder(path['data']['main'] / 'processed_data')
path['data']['processed_data_file'] = path['data']['processed_data'] / 'data.pickle'
path['data']['processed_data_file_target'] = path['data']['processed_data'] / 'target.pickle'
# data from another sources
path['data']['outer_data'] = prepare_folder(path['data']['main'] / 'outer_data')

path['model'] = dict()
path['model']['main'] = prepare_folder(path['main'] / 'model')

for i in range(1, APPROACH_COUNT + 1):
    name = f'approach_{i}'
    path['data'][name] = prepare_folder(path['data']['main'] / name)
    path['model'][name] = prepare_folder(path['model']['main'] / name)



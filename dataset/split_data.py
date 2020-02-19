import glob
import os
import random
import json


def create_data(path):
    def data_labels(path_str):
        files = glob.glob(os.path.join(path, f'**/*_{path_str}.jpeg'))

        return files

    return data_labels('good'), data_labels('bad')


def split_data(save_path):
    def rnd_split(files):
        random.shuffle(files)
        return files[:train_size], files[train_size:]

    path = './data/nailgun/'
    good_files, bad_files = create_data(path)
    train_size = int(len(good_files) * 0.8)
    random.seed(1)

    goods = rnd_split(good_files)
    bads = rnd_split(bad_files)

    splits = {'train': goods[0] + bads[0], 'test': goods[1] + bads[1]}
    for val in splits.values():
        random.shuffle(val)
        
    with open(save_path, 'w') as f:
        json.dump(splits, f)


if __name__ == '__main__':
    split_data('./data/splits.json')

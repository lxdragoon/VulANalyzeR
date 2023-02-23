import os
import time
from itertools import chain
from typing import List
import json

import numpy as np
import tensorflow as tf
from sklearn.metrics import confusion_matrix, f1_score, roc_auc_score
from tensorflow import keras
from tensorflow.compat.v1.keras.layers import CuDNNGRU
from tensorflow.keras.callbacks import (EarlyStopping, LambdaCallback,
                                        ModelCheckpoint)
from tensorflow.keras.layers import (GRU, Bidirectional, Dense, Dropout,
                                     Embedding, Input, LayerNormalization, Conv2D, MaxPooling2D, Flatten,GlobalMaxPooling1D)
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.text import Tokenizer
from tf2_gnn.layers import (GNN, GNNInput)
from xs1.nodes2graph import NodesToGraphRepresentationInput, WeightedSumGraphRepresentation

from typing import Tuple
from tqdm import tqdm
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from tensorflow.keras import backend as K
import requests

from github import Github
g = Github("b03ebbc3514a4e83bada18414a480f757be89afa")

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
physical_devices = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)

def get_commit_json(x):
    return x['bin']['name'].split('\\')[-1].split('-')[0]


devign = pd.read_csv('devign_data.csv')

path_data = os.path.join(os.getcwd(), 'data')
path_devign = os.path.join(path_data, 'devign')
path_bin = os.path.join(path_devign, 'ffmpeg-bins')
path_bin_2 = os.path.join(path_devign, 'ffmpeg-bins-2')
path_filtered = os.path.join(path_devign, 'ffmpeg_filtered')
path_filtered_2 = os.path.join(path_devign, 'ffmpeg_filtered_2')
path_filtered_3 = os.path.join(path_devign, 'ffmpeg_filtered_3')
bin_files = [i for i in os.listdir(path_bin) if i[-4:] == 'json']
bin_files_2 = [i for i in os.listdir(path_bin_2) if i[-4:] == 'json']
#filtered_files = [i for i in os.listdir(path_filtered) if i[-4:] == 'json']


def main():
    data = []
    print('loading binary files')
    for i in tqdm(bin_files):
        with open(os.path.join(path_bin, i)) as f:
            data.append(json.load(f))
    

    #### latest version, remove empty blocks
    ### devign is the csv file, 27k entries in total
    ### data is the json files in ffmpeg-bins, 4936 files in total 
    function_all = devign.func
    data_update = []
    label_update = []
    for index, row in devign[devign['project'] == 'FFmpeg'].iterrows():
        commit_devign = row.commit_id
        for d in [i for i in data if get_commit_json(i) == commit_devign]:
            asm_update = d.copy()
            asm_update['blocks'] = []
            
            function_id = [i['_id'] for i in d['functions'] if row['func'].lower() in str(i['name']).lower()]
            #function_name = [i['name'] for i in d['functions'] if i['_id'] == function_id]
            #asm_update['function_name'] = function_name
            asm_update['devign_index'] = index
            for blk in d['blocks']:
                if blk['ins'] != []:
                    asm_update['blocks'].append(blk)
            if asm_update['blocks'] != []:
                data_update.append(asm_update)
                label_update.append(row['label'])


    ### dump filtered files to json
    ### dump filtered files to json

    if not os.path.exists(os.path.join(path_devign, 'ffmpeg_filtered_3')):
        os.mkdir(os.path.join(path_devign, 'ffmpeg_filtered_3'))
    for ind,i in enumerate(data_update):
        commit = i['bin']['name'].split('\\')[-1].split('-')[0]
        tar = 'bad-' if label_update[ind] == 0 else 'good-'
        temp_name = tar+str(i['devign_index'])+'-'+i['bin']['name'].split('\\')[-1]+'.json'
        with open(os.path.join(path_filtered_3,temp_name), 'w') as outfile:
            json.dump(i, outfile)



if __name__ == "__main__":
    main()
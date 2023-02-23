import os
import time
from itertools import chain
from typing import List

import numpy as np
import tensorflow as tf
import json
from sklearn.metrics import confusion_matrix, f1_score, roc_auc_score
from tensorflow import keras
from tensorflow.compat.v1.keras.layers import CuDNNGRU
from tensorflow.keras.callbacks import (EarlyStopping, LambdaCallback,
                                        ModelCheckpoint)
from tensorflow.keras.layers import (GRU, Bidirectional, Dense, Dropout,
                                     Embedding, Input, LayerNormalization, Conv2D, MaxPooling2D, Flatten)
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.keras.preprocessing.text import Tokenizer
from tf2_gnn.layers import (GNN, GNNInput)
from xs1.nodes2graph import NodesToGraphRepresentationInput, WeightedSumGraphRepresentation

from xs1.dataset import Sample, gen_juliet, gen_ndss, Dataset
from typing import Tuple
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from tensorflow.keras import backend as K

from PIL import Image
import math
import numpy as np
import json
import os


os.environ['CUDA_VISIBLE_DEVICES'] = '0'
physical_devices = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)

model_path = os.path.join(
    'models', os.path.basename(__file__)
)
result_path = os.path.join(
    'results', os.path.basename(__file__)
)
if not os.path.exists(model_path):
    os.makedirs(model_path)
if not os.path.exists(result_path):
    os.makedirs(result_path)

vocab_size = 100000
max_blk = 200
max_seq = 200
max_lnk = 200
maxcnn = 128


def flatten(x): return [list(chain(*s.sequences)) for s in x]


def prep(tknz, d: Dataset, batch_size=96, shuffle=500):

    def _map(x, cwe, gob):
        graphs = x[0].row_lengths(axis=0)
        row = x[0].row_lengths(axis=1)
        offset = tf.reshape(tf.range(graphs), [graphs, 1])
        n2g = tf.RaggedTensor.from_row_lengths(
            tf.ones(tf.reduce_sum(row), dtype=tf.int64), row
        )
        n2g = (n2g * offset).merge_dims(0, 1)
        merged = x[0].merge_dims(0, 1).to_tensor()
        # one hot mask to set the first element to zero
        mask = ~tf.cast(tf.one_hot(0, tf.shape(
            row)[0], dtype=tf.int64), tf.bool)
        # roll the row vector to right once and set the first element as zero
        offset = tf.cumsum(
            tf.roll(row, 1, 0) * tf.cast(mask, tf.int64))
        # add offset
        adj_l = tf.cast(x[1], dtype=tf.int64) + \
            tf.reshape(offset, [tf.shape(row)[0], 1, 1, 1])

        adj_l = adj_l.merge_dims(0, 1).to_tensor()
        return (merged, n2g, adj_l), (cwe, gob)

    sequences = tf.ragged.constant(
        [[c[:max_seq]
          for c in tknz.texts_to_sequences(s.sequences)[:max_blk]]
         for s in d.x])
    adj_list = tf.ragged.constant(
        [[[p for p in b if p[0] < max_blk and p[1] < max_blk][:max_lnk]
          for b in a.adj_list]
         for a in d.x])

    ds = tf.data.Dataset.from_tensor_slices(
        ((sequences, adj_list), d.cwe, d.gob)
    ).batch(batch_size, drop_remainder = True).map(
        _map, num_parallel_calls=tf.data.experimental.AUTOTUNE
    ).cache()

    if shuffle > 0:
        ds = ds.shuffle(shuffle)
    ds = ds.prefetch(
        tf.data.experimental.AUTOTUNE)

    # for (merged, n2g, adj_l), y in ds:
    #     print(y)

    return ds

def prep_conv(d: Dataset, batch_size = 96, shuffle=500):
    
    def _to_image(file):
        with open(file, 'rb') as rf:
            byte_arr = rf.read()
        byte_arr_len = len(byte_arr)
        width = 1024
        if byte_arr_len <= 10000:
            width = 128
        elif byte_arr_len <= 30000:
            width = 128
        elif byte_arr_len <= 60000:
            width = 128
        elif byte_arr_len <= 100000:
            width = 256
        elif byte_arr_len <= 200000:
            width = 512
        elif byte_arr_len <= 500000:
            width = 768
        height = math.floor(byte_arr_len * 1.0 / width)
        arr = np.frombuffer(byte_arr, dtype='B')
        arr = np.reshape(arr[:height*width], [height, width])[:128,:128]
        # 8-bit pixel, black-and-white
        return np.array(Image.fromarray(arr, 'L'), dtype = int)
    
    file_n = [x.file_name[:-9] for x in d.x]
    conv_input = [_to_image(x) for x in file_n]
    conv_input = [tf.convert_to_tensor(x) for x in conv_input]
    conv_input = [tf.reshape(tf.pad(x, [[0,maxcnn-tf.shape(x)[0]], [0,maxcnn-tf.shape(x)[1]]]), [maxcnn, maxcnn, 1]) for x in conv_input]

    ds = tf.data.Dataset.from_tensor_slices(((conv_input), (d.cwe, d.gob))).batch(batch_size).cache()
    if shuffle > 0:
        ds = ds.shuffle(shuffle)
    ds = ds.prefetch(
        tf.data.experimental.AUTOTUNE)
    return ds



def run(ds: Tuple[Dataset, Dataset, Dataset], ds_name,
        epochs=50, batch_size=96, train_op=True, saving_model = False):

    for d, name in zip(ds, ['train', 'valid', 'test']):
        print(name, len(d.x))

    train, valid, test = ds

    

    print('preparing data...')
    #tokenizer = Tokenizer(num_words=vocab_size)
    #tokenizer.fit_on_texts(flatten(train.x))

    ds_train = prep_conv(train, batch_size=batch_size)
    ds_valid = prep_conv(valid, batch_size=batch_size)
    ds_test = prep_conv(test, batch_size=batch_size, shuffle=-1)

    #print(ds_train, ds_valid, ds_test)

    print('building model...')
    model, model_infer = build(len(train.ind2cwe), maxcnn)
    model.summary()

    print('start training...')
    model.compile(
        optimizer=tf.keras.optimizers.Adam(),
        # optimizer=tf.keras.optimizers.Adam(),
        loss=['sparse_categorical_crossentropy', 'binary_crossentropy'],
        metrics=[['SparseCategoricalAccuracy'], ['accuracy', 'AUC']])

    if not os.path.exists(model_path):
        os.makedirs(model_path)
    model_file = os.path.join(
        model_path, '{}-{}.ckpt'.format(
            ds_name, 'gnn-dropout-.2'
        )
    )

    mcp_save = ModelCheckpoint(
        model_file,
        save_best_only=True,
        save_weights_only=True,
        monitor='val_gob_accuracy',
        mode='max')




    ########### ds is ((seq, adj), cwe_label, gob_label) ######

    def eval():
        print()
        print('evaluation')
        print('v0', model.evaluate(ds_valid, verbose=2))
        print('t0', model.evaluate(ds_test, verbose=2))

    def eval_save():
        ds_infer = ds_test.map(lambda x, y: (x,))  # ignore labels
        print()
        print('evaluating probabilities and saving to csv')
        predictions = model_infer.predict(ds_infer)
        print('prediction shapes:', [p.shape for p in predictions])
        onehot = OneHotEncoder()
        onehot.fit(train.cwe.reshape(-1,1))
        cwe_transformed = onehot.transform(
            test.cwe.reshape(-1, 1)).toarray().tolist()

        to_csv = []
        for sample, cwe, gob, pred_cwe, pred_gob in zip(
                test.x, test.cwe.tolist(), test.gob.tolist(),
                predictions[0].tolist(), predictions[1].tolist()):
            to_csv.append([
                sample.file_name,
                cwe,
                gob,
                pred_cwe,
                pred_gob,
                sample.sequences])
        with open('results/{}_{}_prediction.json'.format(
                os.path.basename(__file__),
                ds_name), 'w') as wf:
            json.dump(to_csv, wf)

    monitor = LambdaCallback(
        on_epoch_begin=lambda ep, _: eval() if (ep-1) % 5 == 0 else None
    )

    if train_op:
        if saving_model:
            history = model.fit(ds_train,
                                epochs=epochs,
                                validation_data=ds_valid,
                                callbacks=[mcp_save]
                                )
        else:
            history = model.fit(ds_train,
                            epochs=epochs,
                            validation_data=ds_valid
                            )
        pd.DataFrame(history.history).to_csv(
            os.path.join(result_path, 'log_{}.csv'.format(ds_name))
        )

    model.load_weights(model_file)

    #eval()
    eval_save()


def build(num_cwe, ms):
    conv_in = Input(batch_shape=(None, ms, ms, 1), name='seq_in')

    conv_2d_1 = Conv2D(32, kernel_size = (3,3), activation = 'relu')(conv_in)
    maxpool = MaxPooling2D(pool_size = (2,2))(conv_2d_1)
    conv_2d_3 = Conv2D(32, kernel_size = (3,3), activation = 'relu')(maxpool)
    maxpool_3 = MaxPooling2D(pool_size = (2,2))(conv_2d_3)
    dropout2 = Dropout(0.5)(maxpool_3)
    densed = Flatten()(dropout2)
    densed = Dense(64)(densed)
    dropout3 = Dropout(0.5)(densed)
    cwe = Dense(num_cwe, activation=tf.nn.softmax, name='cwe')(dropout3)
    gob = Dense(1, activation=tf.nn.sigmoid, name='gob')(dropout3)
    md_train = tf.keras.Model(
        inputs=(conv_in), outputs=[cwe, gob])
    md_infer = tf.keras.Model(
    inputs=(conv_in), outputs=[cwe, gob])
    return md_train, md_infer



if __name__ == "__main__":
    #run(gen_ndss(), 'ndss', batch_size=64, epochs=50, train_op=False, saving_model = False)
    run(gen_juliet(), 'juliet', batch_size = 16, epochs=30, train_op=False, saving_model = False)

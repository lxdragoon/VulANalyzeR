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
from tensorflow.keras.layers import (GRU, Bidirectional, Dense, Dropout, Masking,
                                     Embedding, Input, LayerNormalization, GlobalMaxPooling1D)
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.keras.preprocessing.text import Tokenizer
from tf2_gnn.layers import (GNN, GNNInput)
from xs1.nodes2graph import NodesToGraphRepresentationInput, WeightedSumGraphRepresentation

#from xs1.dataset import Sample, gen_juliet, gen_ndss, gen_esh, Dataset, gen_devign
from xs1.dataset_tfr import Sample, gen_juliet, gen_ndss, gen_esh, gen_devign
from typing import Tuple
import pandas as pd
import pickle
from sklearn.preprocessing import OneHotEncoder
from tensorflow.keras import backend as K
from tensorflow.python.ops import math_ops


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

vocab_size = 50000
max_blk = 100
max_seq = 800
max_lnk = 50


def flatten(x): return [list(chain(*s.sequences)) for s in x]


def prep(d, batch_size=96, shuffle=500):


    def _map(x):
        
        debug_inputs = x['sequences'].merge_dims(1, 2).to_tensor()[:,:max_seq]
        # [batch, block, token]
        graphs = x['sequences'].row_lengths(axis=0)
        
        
        row = x['sequences'].row_lengths(axis=1)
        offset = tf.reshape(tf.range(graphs), [graphs, 1])
        n2g = tf.RaggedTensor.from_row_lengths(
            tf.ones(tf.reduce_sum(row), dtype=tf.int64), row
        )
        n2g = (n2g * offset).merge_dims(0, 1)
        merged = x['sequences'].merge_dims(0, 1).to_tensor()[:,:max_seq]
        #merged = x[0].merge_dims(0,1)

        # one hot mask to set the first element to zero
        mask = ~tf.cast(tf.one_hot(0, tf.shape(
            row)[0], dtype=tf.int64), tf.bool)
        # roll the row vector to right once and set the first element as zero
        offset = tf.cumsum(
            tf.roll(row, 1, 0) * tf.cast(mask, tf.int64))
        # add offset
        offset = tf.reshape(offset, [tf.shape(row)[0], 1, 1])
        # adj_l = tf.cast(x['adj_list'], dtype=tf.int64) + offset
            
        # shape = tf.shape(adj_l)
        # adj_l = tf.reshape(adj_l, [shape[0]*shape[1], shape[2],shape[3]])
        adj_l = (x['adj_list']+offset).merge_dims(0, 1).to_tensor()
        return (debug_inputs, n2g, adj_l), (tf.zeros_like(x['cwe']), x['gob'])
    


    ds = d.batch(batch_size).map(
        _map, num_parallel_calls=tf.data.experimental.AUTOTUNE
    )#.take(200)#.cache() 20 000

    
#     if shuffle > 0:
#          ds = ds.shuffle(shuffle)
#     ds = ds.prefetch(
#          tf.data.experimental.AUTOTUNE)

    
    return ds


def run(ds , ds_name,
        epochs=50, batch_size=96, train_op=True, saving_model = False, use_tknz = False):

    # tf.compat.v1.disable_eager_execution()

    train, valid, test = ds
    
    #for i,j in enumerate(train):
    #    print(i)
    #return

    print('building model...')
    model, model_infer = build(len(train.meta['ind2cwe']))
    model.summary()

    print('preparing data...')
    
    # if use_tknz == True:
    #     if os.path.exists(os.path.basename(__file__)+ds_name+'tokenizer.pickle'):
    #         with open(os.path.basename(__file__)+ds_name+'tokenizer.pickle', 'rb') as handle:
    #             tokenizer = pickle.load(handle)
    #     else:
    #         tokenizer = Tokenizer(num_words=vocab_size)
    #         tokenizer.fit_on_texts(flatten(train.x)+flatten(valid.x)+flatten(test.x))
    #         with open(os.path.basename(__file__)+ds_name+'tokenizer.pickle', 'wb') as handle:
    #             pickle.dump(tokenizer, handle, protocol = pickle.HIGHEST_PROTOCOL)
    # else:
    #     tokenizer = None

    if train_op == True:
        ds_train = prep(train, batch_size=batch_size)
    
    
    for i,j in enumerate(ds_train):
        (a,b,c),(d,e) = j
        print('!!!!!!!!!!',a, e)
        if i == 20:
            break
    
    ds_valid = prep( valid, batch_size=batch_size)
    ds_test = prep( test, batch_size=batch_size, shuffle=-1)

    print('start training...')
    model.compile(
#         optimizer=tf.keras.optimizers.RMSprop(momentum=0.0),
        optimizer=tf.keras.optimizers.Adam(),
#         optimizer=tf.keras.optimizers.SGD(.05),
#         optimizer='Adadelta',
        loss=['sparse_categorical_crossentropy', 'binary_crossentropy'],
        metrics=[['SparseCategoricalAccuracy'], ['accuracy', 'BinaryAccuracy', 'AUC']]
        ,loss_weights = [0,1]
        )

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
        for sample, cwe, gob, pred_cwe, pred_gob, attn, seq_attn in zip(
                test.x, test.cwe.tolist(), test.gob.tolist(),
                predictions[0].tolist(), predictions[1].tolist(),
                predictions[2].to_list(), predictions[3].to_list()):
            to_csv.append([
                sample.file_name,
                cwe,
                gob,
                pred_cwe,
                pred_gob,
                attn,
                seq_attn,
                sample.sequences])
        with open('results/{}_{}_prediction.json'.format(
                os.path.basename(__file__),
                ds_name), 'w') as wf:
            json.dump(to_csv, wf)

    monitor = LambdaCallback(
        on_epoch_begin=lambda ep, _: eval() if (ep-1) % 5 == 0 else None
    )
    
    
    print('training')
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

    if not train_op:
        model.load_weights(model_file)

    eval()
    eval_save()


def build(num_cwe):
    seq_in = Input(batch_shape=(None, None), name='seq_in', ragged = False)
    n2g_in = Input(batch_shape=(None,), name='n2g_in', dtype=tf.int32)
    adj_in = Input(batch_shape=(None, 2), name='adj_in', dtype=tf.int32)
    graphs = tf.reduce_max(n2g_in)+1

    num_heads = 10

    emb = Embedding(vocab_size, 128, mask_zero = True)
    embedding = emb(seq_in) # [bs*block, ts, embed_dim(100)]
    mask = tf.cast(math_ops.logical_not(emb.compute_mask(seq_in)), tf.float32) * 1.e9
    gru1 = Bidirectional(GRU(300, return_sequences=False, recurrent_activation = 'sigmoid'))(embedding)
   
#     norm = LayerNormalization()(dense)
    cwe = Dense(num_cwe, activation='softmax', name='cwe')(gru1)
    gob = Dense(1, activation='sigmoid', name='gob')(Dense(256)(gru1)) 
    md_train = tf.keras.Model(
        inputs=(seq_in, n2g_in, adj_in), outputs=[cwe, gob])
    md_infer = md_train
    return md_train, md_infer







if __name__ == "__main__":
    #run(gen_ndss(), 'ndss', batch_size=64, epochs=30, train_op=True, saving_model = False)
    #run(gen_juliet(), 'juliet', batch_size=32, epochs=20, train_op=True, saving_model = False)
    run(gen_devign(), 'devign', batch_size=32, epochs=200, train_op=True, saving_model = False)
    #run_uptrain(gen_devign(), gen_ndss(), 'devign', 'ndss', batch_size=16, epochs=10, train_op=True, saving_model = False)
    #run_test(gen_juliet(), gen_esh('juliet'), 'juliet', 'esh', batch_size = 16, epochs = 1, train_op = False, saving_model = False)
    #run_test(gen_ndss(), gen_esh('ndss'), 'ndss', 'esh', batch_size = 1, epochs = 10, train_op = False, saving_model = False)
    #run(gen_devign(), 'devign', batch_size=32, epochs=30, train_op=True, saving_model = False)
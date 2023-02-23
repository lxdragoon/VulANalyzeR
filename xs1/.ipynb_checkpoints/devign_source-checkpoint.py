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
from xs1.gnn import (GNN, GNNInput)
from xs1.nodes2graph import NodesToGraphRepresentationInput, WeightedSumGraphRepresentation

from xs1.dataset import Sample, gen_juliet, gen_ndss, gen_esh, Dataset, gen_devign
from xs1.data_devign import gen_devign_source
from typing import Tuple
import pandas as pd
import pickle
from sklearn.preprocessing import OneHotEncoder
from tensorflow.keras import backend as K
from tensorflow.python.ops import math_ops

from gensim.models import Word2Vec
from sklearn import preprocessing




if os.environ.get('CUDA_VISIBLE_DEVICES') is None:
    os.environ['CUDA_VISIBLE_DEVICES'] = '2'
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

max_blk = 500
max_seq = 500
max_lnk = 500


def flatten(x): return [list(chain(*s.sequences)) for s in x]


def prep(w2v, le, d: Dataset, batch_size=96, shuffle=500):

    def _map(x, cwe, gob):
        graphs = x[0].row_lengths(axis=0)
        row = x[0].row_lengths(axis=1)
        offset = tf.reshape(tf.range(graphs), [graphs, 1])
        n2g = tf.RaggedTensor.from_row_lengths(
            tf.ones(tf.reduce_sum(row), dtype=tf.int64), row
        )
        n2g = (n2g * offset).merge_dims(0, 1)
        merged = x[0].merge_dims(0, 1).to_tensor()
        #merged = x[0].merge_dims(0,1)

        # one hot mask to set the first element to zero
        mask = ~tf.cast(tf.one_hot(0, tf.shape(
            row)[0], dtype=tf.int64), tf.bool)
        # roll the row vector to right once and set the first element as zero
        offset = tf.cumsum(
            tf.roll(row, 1, 0) * tf.cast(mask, tf.int64))
        # add offset
        adj_l_ast = (tf.cast(x[1][0], dtype=tf.int64) + \
            tf.reshape(offset, [tf.shape(row)[0], 1, 1, 1])).merge_dims(0,1).to_tensor()
        adj_l_cfg = (tf.cast(x[1][1], dtype=tf.int64) + \
            tf.reshape(offset, [tf.shape(row)[0], 1, 1, 1])).merge_dims(0,1).to_tensor()
        adj_l_cdg = (tf.cast(x[1][2], dtype=tf.int64) + \
            tf.reshape(offset, [tf.shape(row)[0], 1, 1, 1])).merge_dims(0,1).to_tensor()
        adj_l_ncs = (tf.cast(x[1][3], dtype=tf.int64) + \
            tf.reshape(offset, [tf.shape(row)[0], 1, 1, 1])).merge_dims(0,1).to_tensor()

        #adj_l = adj_l.merge_dims(0, 1).to_tensor()
        return (merged, n2g, (adj_l_ast,adj_l_cfg, adj_l_cdg, adj_l_ncs)), gob

    sequences = tf.ragged.constant(np.array([np.array([np.append(w2v.wv[i],le.transform([j])) for (i,j) in zip(t.sequences, t.node_type)]) for t in d.x]))
    
    
    adj_list_ast = tf.ragged.constant(
        [[p for p in a.adj_list_ast if p[0] < max_blk and p[1] < max_blk][:max_lnk]
         for a in d.x])
    adj_list_cfg = tf.ragged.constant(
        [[p for p in a.adj_list_cfg if p[0] < max_blk and p[1] < max_blk][:max_lnk]
         for a in d.x])
    adj_list_cdg = tf.ragged.constant(
        [[p for p in a.adj_list_cdg if p[0] < max_blk and p[1] < max_blk][:max_lnk]
         for a in d.x])
    adj_list_ncs = tf.ragged.constant(
        [[p for p in a.adj_list_ncs if p[0] < max_blk and p[1] < max_blk][:max_lnk]
         for a in d.x])

    ds = tf.data.Dataset.from_tensor_slices(
        ((sequences, (adj_list_ast, adj_list_cfg, adj_list_cdg, adj_list_ncs)), d.cwe, d.gob)
    ).batch(batch_size).map(
        _map, num_parallel_calls=tf.data.experimental.AUTOTUNE
    ).cache()

    #if shuffle > 0:
    #    ds = ds.shuffle(shuffle)
    ds = ds.prefetch(
        tf.data.experimental.AUTOTUNE)

    # for (merged, n2g, adj_l), y in ds:
    #     print(y)

    return ds


def run(ds: Tuple[Dataset, Dataset, Dataset], ds_name,
        epochs=50, batch_size=96, train_op=True, saving_model = False):

    #tf.compat.v1.disable_eager_execution()

    for d, name in zip(ds, ['train', 'valid', 'test']):
        print(name, len(d.x))

    train, valid, test = ds
    
    w2v_size = 100
    
    
    
    print('Ratio of vulnerable classes: ', sum(train.gob)/len(train.gob))

    print('building model...')
    model = build(len(train.ind2cwe), w2v_size+1)
    model.summary()

    print('preparing data...')
    
    tokens_all = [i.sequences for i in train.x] +[i.sequences for i in test.x] +[i.sequences for i in valid.x] 
    types_all = [item for sublist in [i.node_type for i in train.x] for item in sublist] +[item for sublist in [i.node_type for i in test.x] for item in sublist] +[item for sublist in [i.node_type for i in valid.x] for item in sublist]


    w2v = Word2Vec(tokens_all, size = w2v_size, window = 5, min_count = 1)
    le = preprocessing.LabelEncoder().fit(types_all)
    
    if train_op == True:
        ds_train = prep(w2v, le, train, batch_size=batch_size)
        #for d in ds_train:
        #    print(tf.reduce_max(d[0][1])+1)
        
    ds_valid = prep(w2v, le, valid, batch_size=batch_size)
    ds_test = prep(w2v, le, test, batch_size=batch_size, shuffle=-1)

    print('start training...')
    model.compile(
        optimizer=tf.keras.optimizers.RMSprop(momentum=0.0),
        #optimizer=tf.keras.optimizers.Adam(),
        loss='binary_crossentropy',
        metrics=['accuracy', 'AUC']
        #,loss_weights = [0,1]
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
        predictions = model.predict(ds_infer)
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


def build(num_cwe, w2v_size):
    seq_in = Input(batch_shape=(None, w2v_size), name='seq_in')
    n2g_in = Input(batch_shape=(None,), name='n2g_in', dtype=tf.int32)
    adj_in = Input(batch_shape=(None, None, 2), name='adj_in', dtype=tf.int32)
    graphs = tf.reduce_max(n2g_in)+1

    num_heads = 10

    flat_adj = tf.reshape(adj_in, ([-1, 2]))

    gnn_input = GNNInput(
        node_features=seq_in,
        adjacency_lists=(flat_adj,),
        node_to_graph_map=n2g_in,
        num_graphs=graphs)

    params = GNN.get_default_hyperparameters()
    params['num_layers'] = 6
    params['hidden_dim'] = 200
    params['global_exchange_dropout_rate'] = 0
    params['global_exchange_weighting_fun'] = "average"
    #params['global_exchange_weighting_fun'] = "sum"
    gnn = GNN(params)(gnn_input)
    gnn = Dropout(0.3)(gnn)

    reducer_input = NodesToGraphRepresentationInput(
        node_embeddings=gnn,
        node_to_graph_map=n2g_in,
        num_graphs=graphs)

    wsgr, attn = WeightedSumGraphRepresentation(
        16*10, 16, weighting_fun='softmax', name='weightedSum')(reducer_input)

    reduced = tf.reshape(wsgr, [graphs, 16*10])

    attn = tf.reduce_max(attn, axis = -1)
    attn = tf.keras.layers.Lambda(
        lambda x: tf.RaggedTensor.from_value_rowids(x[0], x[1], validate=False))(
            [attn, n2g_in])


    dense = Dense(40, activation=tf.nn.sigmoid)(reduced)

    norm = LayerNormalization()(dense)
    cwe = Dense(num_cwe, activation=tf.nn.softmax, name='cwe')(norm)
    gob = Dense(1, activation=tf.nn.sigmoid, name='gob')(norm) 
    md_train = tf.keras.Model(
        inputs=(seq_in, n2g_in, adj_in), outputs=gob)
    return md_train







if __name__ == "__main__":
    run(gen_devign_source(), 'devign_source', batch_size=32, epochs=30, train_op=True, saving_model = False)
    #run(gen_ndss(), 'ndss', batch_size=80, epochs=30, train_op=True, saving_model = False)
    #run(gen_juliet(), 'juliet', batch_size=12, epochs=20, train_op=False, saving_model = False)
    #run(gen_devign(), 'devign', batch_size= 12, epochs=30, train_op=True, saving_model = False)
    #run_uptrain(gen_devign(), gen_ndss(), 'devign', 'ndss', batch_size=16, epochs=10, train_op=True, saving_model = False)
    #run_test(gen_juliet(), gen_esh('juliet'), 'juliet', 'esh', batch_size = 16, epochs = 1, train_op = False, saving_model = False)
    #run_test(gen_ndss(), gen_esh('ndss'), 'ndss', 'esh', batch_size = 1, epochs = 10, train_op = False, saving_model = False)

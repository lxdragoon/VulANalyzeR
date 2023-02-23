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
max_seq = 400
max_lnk = 100


def flatten(x): return [list(chain(*s.sequences)) for s in x]


def prep(d, batch_size=96, shuffle=500):


    def _map(x):
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
        adj_l = (x['adj_list']+offset)[:,:max_lnk,:].merge_dims(0, 1).to_tensor()
        return (merged, n2g, adj_l), (tf.zeros_like(x['cwe']), x['gob'])
    


    ds = d.batch(batch_size).map(
        _map, num_parallel_calls=tf.data.experimental.AUTOTUNE
    )#.take(50)#.cache()

    
    #if shuffle > 0:
    #     ds = ds.shuffle(shuffle)
    #ds = ds.prefetch(
    #     tf.data.experimental.AUTOTUNE)

    
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
    
    
    #for i,j in enumerate(ds_train):
    #    (a,b,c),(d,e) = j
    #    print('!!!!!!!!!!',a.shape, b.shape, c.shape)
    #    if i == 0:
    #        return
    
    ds_valid = prep( valid, batch_size=batch_size)
    ds_test = prep( test, batch_size=batch_size, shuffle=-1)

    print('start training...')
    model.compile(
        #optimizer=tf.keras.optimizers.RMSprop(momentum=0.0),
        optimizer=tf.keras.optimizers.Adam(),
        loss=['sparse_categorical_crossentropy', 'binary_crossentropy'],
        metrics=[['SparseCategoricalAccuracy'], ['accuracy']]
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
    gru1 = Bidirectional(GRU(300, return_sequences=True, recurrent_activation = 'sigmoid'))(embedding)
    gru1 = Masking()(gru1)
    gru2 = Bidirectional(GRU(300, return_sequences=True, recurrent_activation = 'sigmoid', name = 'gru2'))(gru1) # [bs*block, ts, gru_dim(100*2)]

    
    key = Dense(num_heads, name ='attention_key', use_bias = False)(gru2)  # [bs*block, ts, heads]
    key -= tf.expand_dims(mask, -1)
    score = tf.expand_dims(tf.keras.activations.softmax(key, axis = 1), -1) # [bs*block, ts, heads] -> [bs*block, ts, heads, 1]
    attention_weights = tf.reshape(score, tf.shape(score)[:-1]) # [bs*block, ts, heads]
    value = Dense(200, name = 'attention_value', use_bias = False)(gru2) # [bs*block, ts, 200]
    value = tf.reshape(value, [tf.shape(value)[0], tf.shape(value)[1], num_heads, int(200/num_heads)]) # [bs*block, ts, heads, 200/heads]
    attned = score*value # [bs*block, ts, heads, 200/heads]
    attention_vector = tf.reduce_sum(attned, axis = 1) # [bs*block, heads, 200/heads]
    attention_vector = tf.reshape(attention_vector, (tf.shape(attention_vector)[0], 200)) # [bs*block, 200]
    attention_vector = Dropout(0.3)(attention_vector)


    flat_adj = adj_in #tf.reshape(adj_in, ([-1, 2]))

    gnn_input = GNNInput(
        node_features=attention_vector,
        adjacency_lists=(flat_adj,),
        node_to_graph_map=n2g_in,
        num_graphs=graphs)

    params = GNN.get_default_hyperparameters()
    params['hidden_dim'] = 64
    params['global_exchange_dropout_rate'] = 0
    gnn = GNN(params)(gnn_input)
    gnn = Dropout(0.3)(gnn)

    reducer_input = NodesToGraphRepresentationInput(
        node_embeddings=gnn, # steven
        node_to_graph_map=n2g_in,
        num_graphs=graphs)

    wsgr, attn = WeightedSumGraphRepresentation(
        16*num_heads, 16, weighting_fun='softmax', name='weightedSum')(reducer_input)

    reduced = tf.reshape(wsgr, [graphs, 16*num_heads])

    attn = tf.reduce_max(attn, axis = -1)
    attn = tf.keras.layers.Lambda(
        lambda x: tf.RaggedTensor.from_value_rowids(x[0], x[1], validate=False))(
            [attn, n2g_in])

    batch = tf.shape(attention_weights)[0]
    attn_seq = tf.reduce_max(attention_weights, axis = -1)
    to_pad = tf.zeros([batch, max_seq - tf.shape(attention_weights)[1]])
    attention_weights_pad = tf.concat([attn_seq, to_pad], -1)
    attention_weights_out = tf.keras.layers.Lambda(
        lambda x: tf.RaggedTensor.from_value_rowids(x[0], x[1], validate=False))(
            [attention_weights_pad, n2g_in])

    dense = Dense(64)(reduced)

    norm = LayerNormalization()(dense)
    cwe = Dense(num_cwe, activation='softmax', name='cwe')(norm)
    gob = Dense(1, activation='sigmoid', name='gob')(norm) 
    md_train = tf.keras.Model(
        inputs=(seq_in, n2g_in, adj_in), outputs=[cwe, gob])
    md_infer = tf.keras.Model(
        inputs=(seq_in, n2g_in, adj_in), outputs=[cwe, gob, attn, attention_weights_out])
    return md_train, md_infer


def run_test(ds, test_ds, ds_name, ds_name_test,
        epochs=50, batch_size=96, train_op=True, saving_model = False):

    #tf.compat.v1.disable_eager_execution()


#     for d, name in zip(ds, ['train', 'valid', 'test']):
#         print(name, len(d.x))

    train, valid, test = ds
    if isinstance(test_ds, tuple):
        test1, test2, test3 = test_ds
    else:
        test1 = test_ds

    print('building model...')
    model, model_infer = build(len(train.meta['ind2cwe']))
    model.summary()

    print('preparing data...')
    ##ds_test.map()
    ds_valid = prep(valid, batch_size=batch_size)
    ds_test = prep(test1, batch_size=batch_size, shuffle=-1)

    print('start training...')
    model.compile(
        #optimizer=tf.keras.optimizers.RMSprop(momentum=0.),
        optimizer=tf.keras.optimizers.Adam(),
        loss=['sparse_categorical_crossentropy', 'binary_crossentropy'],
        metrics=[['SparseCategoricalAccuracy'], ['accuracy', 'AUC']])


    if not os.path.exists(model_path):
        os.makedirs(model_path)
    model_file = os.path.join(
        model_path, '{}-{}.ckpt'.format(
            ds_name, 'gnn-dropout-.2'
        )
    )
    ########### ds is ((seq, adj), cwe_label, gob_label) ######

    def eval():
        print()
        print('evaluation')
        #print('t0', model.evaluate(ds_valid, verbose=2))
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
            ds_test.cwe.reshape(-1, 1)).toarray().tolist()

        to_csv = []
        for sample, cwe, gob, pred_cwe, pred_gob, attn, seq_attn in zip(
                ds_test.x, ds_test.cwe.tolist(), ds_test.gob.tolist(),
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
                ds_name_test), 'w') as wf:
            json.dump(to_csv, wf)


    model.load_weights(model_file)

    eval()
    eval_save()



def run_uptrain(ds, ndss_ds, ds_up_name, ds_ori_name,
        epochs=50, batch_size=96, train_op=True, saving_model = False):

    #tf.compat.v1.disable_eager_execution()


    train, valid, test = ds
    ndss_train, ndss_valid, ndss_test = ndss_ds

    print('building model...')
    model, model_infer = build(len(ndss_train.meta['ind2cwe']))
    model.summary()
    ds_name = ds_up_name+ds_ori_name


    print('preparing data...')


    if train_op == True:
        ds_train = prep(train, batch_size=batch_size)
    ds_valid = prep(valid, batch_size=batch_size)
    ds_test = prep(test, batch_size=batch_size, shuffle=-1)

    print('start training...')
    model.compile(
        optimizer=tf.keras.optimizers.RMSprop(momentum=0.),
        #optimizer=tf.keras.optimizers.Adam(),
        loss=['sparse_categorical_crossentropy', 'binary_crossentropy'],
        metrics=[['SparseCategoricalAccuracy'], ['accuracy', 'AUC']]
        , loss_weights = [0,1]
        )
    
    
    

    if not os.path.exists(model_path):
        os.makedirs(model_path)
    model_file = os.path.join(
        model_path, '{}-{}.ckpt'.format(
            ds_ori_name, 'gnn-dropout-.2'
        )
    )

    model.load_weights(model_file)
    ########### ds is ((seq, adj), cwe_label, gob_label) ######

    mcp_save = ModelCheckpoint(
        model_file,
        save_best_only=True,
        save_weights_only=True,
        monitor='val_gob_accuracy',
        mode='max')
    
    def eval():
        print()
        print('evaluation')
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
            ds_test.cwe.reshape(-1, 1)).toarray().tolist()

        to_csv = []
        for sample, cwe, gob, pred_cwe, pred_gob, attn, seq_attn in zip(
                ds_test.x, ds_test.cwe.tolist(), ds_test.gob.tolist(),
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
                ds_name_test), 'w') as wf:
            json.dump(to_csv, wf)

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

    #if saving_model:
    #    model.load_weights(model_file)

    eval()
    eval_save()





if __name__ == "__main__":
    ##run(gen_ndss(), 'ndss', batch_size=32, epochs=20, train_op=True, saving_model = True)
    run(gen_juliet(), 'juliet', batch_size=32, epochs=20, train_op=True, saving_model = False)
    ##run(gen_devign(), 'devign', batch_size=6, epochs=80, train_op=True, saving_model = False)
    #run_uptrain(gen_devign(), gen_ndss(), 'devign', 'ndss', batch_size=16, epochs=10, train_op=True, saving_model = False)
    #run_uptrain(gen_esh(), gen_ndss(), 'esh', 'ndss', batch_size=8, epochs=10, train_op=True, saving_model = False)
    #run_test(gen_juliet(), gen_esh('juliet'), 'juliet', 'esh', batch_size = 16, epochs = 1, train_op = False, saving_model = False)
    #run_test(gen_ndss(), gen_esh(), 'ndss', 'esh', batch_size = 16, epochs = 10, train_op = False, saving_model = False)
    #run(gen_devign(), 'devign', batch_size=32, epochs=30, train_op=True, saving_model = False)
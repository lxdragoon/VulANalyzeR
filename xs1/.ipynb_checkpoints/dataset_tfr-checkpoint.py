import os
import numpy as np
from sklearn import model_selection
from itertools import chain
from typing import NamedTuple, List, Dict, Tuple
from tqdm import tqdm
import gzip
import json
import pickle
import re
import pandas as pd
from pathlib import Path
from xs1.tokenizers.tokenizer import blk2seq, get_tokenizer_xl,get_tokenizer_sm
from xs1.tfr import read_objs_trf_ds, write_objs_as_tfr
import sentencepiece as spm
from functools import partial
import ast


vocab_size = 50000
max_blk = 100
max_seq = 400
max_lnk = 100

sp_model = os.path.join(os.path.join('xs1', 'tokenizers'),
                        'tokenizer_10000000_30000.sp.model')
sp = spm.SentencePieceProcessor()
sp.Load(sp_model)


class AttrDict(dict):
    """ Dictionary subclass whose entries can be accessed by attributes
        (as well as normally).
    """

    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self

    @staticmethod
    def from_nested_dict(data):
        """ Construct nested AttrDicts from nested dictionaries. """
        if isinstance(data, dict):
            return AttrDict({key: AttrDict.from_nested_dict(data[key])
                             for key in data})
        if isinstance(data, list):
            return [AttrDict.from_nested_dict(d) for d in data]
        return data


def train_valid_test_split(*args, splits=[.6, .3, .1]):
    rnd1 = model_selection.train_test_split(
        *args, test_size=splits[1]+splits[2],
        random_state=0,
    )
    first = [rnd1[i*2] for i in range(0, len(args))]
    second = [rnd1[i*2+1] for i in range(0, len(args))]
    rnd2 = model_selection.train_test_split(
        *second, test_size=splits[2]/(splits[1]+splits[2]),
        random_state=0,
    )
    second = [rnd2[i*2] for i in range(0, len(args))]
    third = [rnd2[i*2+1] for i in range(0, len(args))]
    return first, second, third


path_data = 'data-tfr'
path_ndss = os.path.join(path_data, 'ndss')
path_juliet = os.path.join(path_data, 'juliet')
path_esh = os.path.join(path_data, 'esh')
#path_devign = os.path.join(
#    os.path.join(path_data, 'devign'), 'ffmpeg_filtered')
#path_devign = os.path.join(
#    os.path.join(path_data, 'devign'), 'ffmpeg_filtered_2')
# path_devign = os.path.join(os.path.join(
#     path_data, 'devign'), 'ffmpeg_filtered_3')

path_devign = os.path.join(path_data, 'devign-v2')
#path_devign = os.path.join(os.path.join(path_data, 'devign-v2'), 'filtered')

oprs_filter = ('loc_', 'sub_', 'arg_', 'var_',
               'word_', 'off_', 'locret_', 'flt_', 'dbl_', 'param_', 'local_')


esh_flags = {
    'dtls1_process_heartbeat': 'cve-2014-0160',
    'initialize_shell_variable': 'cve-2014-6271',
    'fdctrl_handle_drive_specification_command': 'cve-2015-3456',
    '__configure': 'cve-2014-9295',
    '__parse_and_execute': 'cve-2014-7169',
    'snmp_usm_password_to_key_sha1': 'cve-2011-0444',
    '__ftp_retrieve_glob': 'cve-2014-4877',
    '__ff_rv34_decode_init_thread_copy': 'cve-2015-6826'
}


class Sample(NamedTuple):
    sequences: List[str]
    adj_matrix: List[int]
    adj_list: List[int]
    file_name: str
    cwe: int
    gob: int


def json2sample_old(js_file) -> Sample:
    with open(js_file) as rf:
        obj = json.load(rf)

    obj = AttrDict.from_nested_dict(obj)

    def blk2seq(blk):
        for i in blk.ins:
            for tkn in [i.mne] + i.oprs:
                for p in re.split(r'[+\-*\\\[\]:()\s]', tkn.lower()):
                    if not p.startswith(oprs_filter) and len(p) > 0:
                        yield p

    sequences = [['<blk>']+list(blk2seq(b))
                 for b in obj.blocks]
    id2ind = {b._id if hasattr(b, '_id') else b.addr_start: ind for ind, b in enumerate(obj.blocks)}

    mat = np.zeros((len(id2ind), len(id2ind)))
    adj_l = []
    for ind, b in enumerate(obj.blocks):
        blk_adj_l = []
        for c in b.calls:
            if c in id2ind:
                mat[ind, id2ind[c]] = 1
                blk_adj_l.append((ind, id2ind[c]))
        adj_l.append(blk_adj_l)

    return Sample(sequences, mat, adj_l, str(js_file))


def json2sample(obj, js_file, fn_label, cwe2ind) -> Sample:

    obj = AttrDict.from_nested_dict(obj)

    sp = get_tokenizer_xl(vocab_size = vocab_size)
    
    sequences = [list(blk2seq(b, sp, flatten=True))
                 for b in obj.blocks][:max_blk]
    id2ind = {b._id if hasattr(b, '_id') else b.addr_start: ind for ind, b in enumerate(obj.blocks)}

    mat = np.zeros((len(id2ind), len(id2ind)))
    adj_l = []
    for ind, b in enumerate(obj.blocks):
        #blk_adj_l = []
        all_ref = []
        for i in b['ins']:
            cr = i.get('cr', [])
            dr = i.get('dr', [])
            all_ref.extend(cr)
            all_ref.extend(dr)

        for c in all_ref:
            if c in id2ind:
                mat[ind, id2ind[c]] = 1
                adj_l.append([ind, id2ind[c]])
    
        for c in b.calls:
            if c in id2ind:
                mat[ind, id2ind[c]] = 1
                adj_l.append([ind, id2ind[c]])
    
    c0, c1 = fn_label(str(js_file))
    #return Sample(sequences, mat, adj_l, str(js_file)), c0, c1
    return {
        'sequences': sequences,
        'adj_matrix': mat.tolist(),
        'adj_list': adj_l, # [None, 2]
        'file_name': str(js_file),
        'cwe': c0,
        'gob': c1
    }

def json2sample_devign(obj, js_file, fn_label, cwe2ind) -> Sample:
        

    obj = AttrDict.from_nested_dict(obj)
    target_func = str(js_file).split('.')[-5]
    target_addr = [i for i in obj['functions'] if i['name'] == target_func]
    if len(target_addr) > 0: #and ('opt2' in str(js_file) or 'opt3' in str(js_file)):
        target_addr = target_addr[0]['addr_start']
        blocks = []
        all_ref = []
        for b in obj['blocks']:
            if b['addr_f'] == target_addr:
                blocks.append(b)
                for i in b['ins']:
                    cr = i.get('cr', [])
                    dr = i.get('dr', [])
                    all_ref.extend(cr)
                    all_ref.extend(dr)
        all_ref = set(all_ref)
        for b in obj['blocks']:
            if b['addr_f'] != target_addr and b['addr_start'] in all_ref:
                blocks.append(b)


        #obj.blocks = [i for i in obj['blocks'] if i['addr_f'] == target_addr or i['addr_start'] == target_addr or i['addr_end'] == target_addr or target_addr in i['calls']]
        obj.blocks = [i for i in obj['blocks'] if i['addr_f'] == target_addr]
    else:
        return
    
    #model = os.path.join(os.path.join('xs1', 'tokenizers'), 'tokenizer_10000000_30000.sp.model')
    #sp = spm.SentencePieceProcessor()
    #sp.Load(model)
    sp = get_tokenizer_sm(vocab_size = vocab_size)
    #sp = get_tokenizer_xl(vocab_size = vocab_size)
    
    sequences = [list(blk2seq(b, sp, flatten=True))
                 for b in obj.blocks][:max_blk]
    id2ind = {b._id if hasattr(b, '_id') else b.addr_start: ind for ind, b in enumerate(obj.blocks)}

    mat = np.zeros((len(id2ind), len(id2ind)))
    adj_l = []
    for ind, b in enumerate(obj.blocks):
        #blk_adj_l = []
        all_ref = []
        for i in b['ins']:
            cr = i.get('cr', [])
            dr = i.get('dr', [])
            all_ref.extend(cr)
            all_ref.extend(dr)

        for c in all_ref:
            if c in id2ind:
                mat[ind, id2ind[c]] = 1
                adj_l.append([ind, id2ind[c]])
    
        for c in b.calls:
            if c in id2ind:
                mat[ind, id2ind[c]] = 1
                adj_l.append([ind, id2ind[c]])
    
    c0, c1 = fn_label(str(js_file))
    #return Sample(sequences, mat, adj_l, str(js_file)), c0, c1
    return {
        'sequences': sequences,
        'adj_matrix': mat.tolist(),
        'adj_list': adj_l, # [None, 2]
        'file_name': str(js_file),
        'cwe': c0,
        'gob': c1
    }


def gen(dir, fn_label, trunk_size=50):

    cache = os.path.join(dir, 'tfrs')
    if os.path.exists(cache):
        full = os.path.join(cache, 'full')
        train = os.path.join(cache, 'train')
        valid = os.path.join(cache, 'valid')
        test = os.path.join(cache, 'test')
        if os.path.exists(full):
            return read_objs_trf_ds(full)
        else:
            return read_objs_trf_ds(train), read_objs_trf_ds(valid), read_objs_trf_ds(test)

    all_fs = list(Path(dir).glob('**/*.asm.json'))

    cwe, _ = list(zip(*[fn_label(f) for f in all_fs]))
    ind2cwe = sorted(set(cwe))
    cwe2ind = {c: i for i, c in enumerate(ind2cwe)}

    def write(selected_js, name):
        if 'devign' in dir:
            return write_objs_as_tfr(
                selected_js, os.path.join(cache, name),
                obj_mapper=partial(
                    json2sample_devign, fn_label=fn_label, cwe2ind=cwe2ind),
                additional_meta={'ind2cwe': ind2cwe},
                trunk_size=trunk_size, 
                num_workers=16)
        else:
            return write_objs_as_tfr(
                selected_js, os.path.join(cache, name),
                obj_mapper=partial(
                    json2sample, fn_label=fn_label, cwe2ind=cwe2ind),
                additional_meta={'ind2cwe': ind2cwe},
                trunk_size=trunk_size, 
                num_workers=16)
    
    #### change this if testing on esh
    
    #if dir[-3:] == 'esh':
    if dir[-3:] == '???':
        return write(all_fs, 'full')
    else:
        train, valid, test = train_valid_test_split(all_fs)
        train, valid, test = write(train[0], 'train'), write(
            valid[0], 'valid'), write(test[0], 'test')
        return train, valid, test


def label_ndss(js_file: str):
    js_file = os.path.basename(js_file)
    gob = 0 if 'good' in js_file else 1
    cwe = int(js_file.split('_')[0].replace('cwe', ''))
    return cwe, gob


def gen_ndss():
    return gen(path_ndss, fn_label=label_ndss)


def label_julliet(js_file: str):
    js_file = os.path.basename(js_file)
    gob = 0 if 'good' in js_file else 1
    cwe = int(js_file.split('_')[0].replace('CWE', ''))
    return cwe, gob


def gen_juliet():
    return gen(path_juliet, fn_label=label_julliet)


def label_esh(js_file: str):
    js_file = os.path.basename(js_file)
    gob = 0
    cwe = 121
    # if ds_name == 'ndss':
    #     cwe = 119
    # else:
    #     cwe = 121
    for i in esh_flags.keys():
        if i in js_file:
            gob = 1
            return cwe, gob

    return cwe, gob


def gen_esh():
    return gen(path_esh, fn_label=label_esh)


def label_devign(js_file: str):
    js_file = os.path.basename(js_file)
    gob = 0 if 'good' in js_file else 1
    cwe = 121 if 'good' in js_file else 1
    return cwe, gob


def gen_devign():
    return gen(path_devign, fn_label=label_devign, trunk_size=50)


def gen_unlabeled(path):
    def label(js_file: str):
        return 0, 0
    return gen(os.path.join(path_data, path), fn_label=label)


if __name__ == "__main__":
    # gen_ndss()
    # train, valid, test = gen_juliet()
    # gen_esh()
    # print(train.meta['ind2cwe'])
#     for d in train:
#         print(d)
#         break
    tr,vl,te = gen_devign()
    for d in tr:
        print(d)
        break
    # gen_devign()

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
from xs1.tokenizers.tokenizer import blk2seq,get_tokenizer_sm, get_tokenizer_xl
import sentencepiece as spm
import ast
from concurrent.futures import ProcessPoolExecutor
from functools import partial


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


def train_valid_test_split(*args, splits=[.8, .1, .1]):
    rnd1 = model_selection.train_test_split(
        *args, test_size=splits[1]+splits[2],
        random_state=0, stratify=args[-1]
    )
    first = [rnd1[i*2] for i in range(0, len(args))]
    second = [rnd1[i*2+1] for i in range(0, len(args))]
    rnd2 = model_selection.train_test_split(
        *second, test_size=splits[2]/(splits[1]+splits[2]),
        random_state=0, stratify=second[-1]
    )
    second = [rnd2[i*2] for i in range(0, len(args))]
    third = [rnd2[i*2+1] for i in range(0, len(args))]
    return first, second, third


path_data = 'data'
path_ndss = os.path.join(path_data, 'ndss')
path_juliet = os.path.join(path_data, 'juliet')
path_esh = os.path.join(path_data, 'esh')
#path_devign = os.path.join(os.path.join(path_data, 'devign'), 'ffmpeg_filtered')
#path_devign = os.path.join(os.path.join(path_data, 'devign'), 'ffmpeg_filtered_2')
#path_devign = os.path.join(os.path.join(path_data, 'devign'), 'ffmpeg_filtered_3')
#path_devign = os.path.join(path_data, 'devign-v2')
path_devign = os.path.join(os.path.join(path_data, 'devign-v2'), 'filtered')

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


class Dataset(NamedTuple):
    x: List[Sample]
    cwe: np.ndarray
    gob: np.ndarray
    ind2cwe: dict


def json2sample_old(js_file, fn_label=None) -> Sample:
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

    c0, c1 = fn_label(str(js_file))
    return Sample(sequences, mat, adj_l, str(js_file)), c0, c1
    # return Sample(sequences, mat, adj_l, str(js_file))

def json2sample(js_file, fn_label=None) -> Sample:
    with open(js_file) as rf:
        obj = json.load(rf)

    obj = AttrDict.from_nested_dict(obj)


    model = os.path.join(os.path.join('xs1', 'tokenizers'), 'tokenizer_10000000_30000.sp.model')
    # sp = spm.SentencePieceProcessor()
    # sp.Load(model)
    
    sp = get_tokenizer_xl()
    
    sequences = [list(blk2seq(b, sp, flatten=True))
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

    c0, c1 = fn_label(str(js_file))
    return Sample(sequences, mat, adj_l, str(js_file)), c0, c1


def json2sample_devign(js_file, fn_label=None) -> Sample:
        
    with gzip.open(js_file, 'rb') as rf:
        obj = ast.literal_eval(rf.read().decode('utf-8'))

    obj = AttrDict.from_nested_dict(obj)
    target_func = str(js_file).split('.')[-5]
    target_addr = [i for i in obj['functions'] if i['name'] == target_func]
    if len(target_addr) > 0:
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


        obj.blocks = [i for i in obj['blocks'] if i['addr_f'] == target_addr]
    else:
        return None
    
    # model = os.path.join(os.path.join('xs1', 'tokenizers'), 'tokenizer_10000000_30000.sp.model')
    #sp = get_tokenizer_sm()
    sp = get_tokenizer_xl()
    
    sequences = [list(blk2seq(b, sp, flatten=True))
                 for b in obj.blocks]
    id2ind = {b._id if hasattr(b, '_id') else b.addr_start: ind for ind, b in enumerate(obj.blocks)}

    mat = np.zeros((len(id2ind), len(id2ind)))
    adj_l = []
    for ind, b in enumerate(obj.blocks):
        blk_adj_l = []
        all_ref = []
        for i in b['ins']:
            cr = i.get('cr', [])
            dr = i.get('dr', [])
            all_ref.extend(cr)
            all_ref.extend(dr)

        for c in all_ref:
            if c in id2ind:
                mat[ind, id2ind[c]] = 1
                blk_adj_l.append((ind, id2ind[c]))
        adj_l.append(blk_adj_l)
    
    c0, c1 = fn_label(str(js_file))
    return Sample(sequences, mat, adj_l, str(js_file)), c0, c1




def gen(dir, fn_label, test=0.2, save=True) -> Tuple[Dataset, Dataset, Dataset]:

    sp = get_tokenizer_sm()

    if save:
        cache = os.path.join(dir, 'all.json.gz')
        if os.path.exists(cache):
            with gzip.open(cache, 'rb') as rf:
                return pickle.load(rf)

    #all_fs = list(Path(dir).glob('**/*.asm.json')) if not clear'devign' in dir else list(Path(dir).glob('**/*.json.gz'))
    all_fs = list(Path(dir).glob('**/*.json'))
    #mapper = json2sample_devign if 'devign' in dir else json2sample
    mapper = json2sample_old
    # mapper = json2sample

    ls = [mapper(s, fn_label) for s in tqdm(all_fs)]
    ls = [s for s in ls if s]
    x, cwe, gob = list(zip(*ls))

    ind2cwe = sorted(set(cwe))
    cwe2ind = {c: i for i, c in enumerate(ind2cwe)}
    cwe = [cwe2ind[c] for c in cwe]

    gob = np.array(gob)  # good or bad
    cwe = np.array(cwe)  # cwe code

    print('# label gob')
    print(pd.Series(gob).astype('category').describe())
    print('# label cwe')
    print(pd.Series(cwe).astype('category').describe())
    print('# link count')
    print(pd.Series([len(a.adj_matrix) for a in x]).describe())
    print('# block count')
    print(pd.Series([len(a.sequences) for a in x]).describe())
    print('# seq len')
    print(pd.Series([len(b)
                     for a in x for b in a.sequences]).describe())

    if dir[-3:] == 'esh':
        ds = Dataset(x,cwe,gob, ind2cwe)
        if save:
            with gzip.open(cache, 'wb') as wf:
                pickle.dump(ds, wf)
        return ds

    train, valid, test = train_valid_test_split(x, cwe, gob)
    ds = Dataset(*train, ind2cwe), Dataset(
        *valid, ind2cwe), Dataset(*test, ind2cwe)
    if save:
        with gzip.open(cache, 'wb') as wf:
            pickle.dump(ds, wf)
    return ds


def gen_ndss():
    def label(js_file: str):
        js_file = os.path.basename(js_file)
        gob = 0 if 'good' in js_file else 1
        cwe = int(js_file.split('_')[0].replace('cwe', ''))
        return cwe, gob
    return gen(path_ndss, fn_label=label)


def gen_juliet():
    def label(js_file: str):
        js_file = os.path.basename(js_file)
        gob = 0 if 'good' in js_file else 1
        cwe = int(js_file.split('_')[0].replace('CWE', ''))
        return cwe, gob
    return gen(path_juliet, fn_label=label)


def gen_esh(ds_name):
    def label(js_file: str):
        js_file = os.path.basename(js_file)
        gob = 0
        if ds_name == 'ndss':
            cwe = 119
        else:
            cwe = 121
        for i in esh_flags.keys():
            if i in js_file:
                gob = 1
                return cwe, gob
                
        return cwe, gob
    return gen(path_esh, fn_label=label)

def gen_devign():
    def label(js_file: str):
        js_file = os.path.basename(js_file)
        gob = 0 if 'good' in js_file else 1
        cwe = 121 if 'good' in js_file else 1
        return cwe, gob
    return gen(path_devign, fn_label=label)


def gen_unlabeled(path):
    def label(js_file: str):
        return 0, 0
    return gen(os.path.join(path_data, path), fn_label=label, save=False)


if __name__ == "__main__":
    #gen_ndss()
    #gen_juliet()
    #gen_devign()
    gen_esh('ndss')

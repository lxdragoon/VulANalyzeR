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

path_data = 'data'
path_devign = os.path.join(path_data, 'devign')
path_devign_source = os.path.join(path_devign, 'source_devign')



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
    
    
class Sample(NamedTuple):
    sequences: List[str]
    node_type: str
    adj_list_ast: List[int]
    adj_list_cfg: List[int]
    adj_list_cdg: List[int]
    adj_list_ncs: List[int]
    file_name: str


class Dataset(NamedTuple):
    x: List[Sample]
    cwe: np.ndarray
    gob: np.ndarray
    ind2cwe: dict
        

def json2sample(js_file) -> Sample:
    with open(js_file) as rf:
        obj = json.load(rf)

    obj = AttrDict.from_nested_dict(obj)
    
    sequences = [i[2]['CODE'] for i in obj['vertices']]
    
    node_type = [i[1] for i in obj['vertices']]
    
    id2ind = {b[0]: ind for ind, b in enumerate(obj.vertices)}
    
    mat = np.zeros((len(obj.vertices), len(obj.vertices)))
    adj_l_ast = []
    adj_l_cfg = []
    adj_l_cdg = []
    adj_l_ncs = []
    for ind, b in enumerate(obj.edges):
        if b[0] == 'AST':
            adj_l_ast.append((id2ind[b[1]],id2ind[b[2]]))
        if b[0] == 'CFG':
            adj_l_cfg.append((id2ind[b[1]],id2ind[b[2]]))
        if b[0] == 'CDG':
            adj_l_cdg.append((id2ind[b[1]],id2ind[b[2]]))
        if b[0] == 'NCS':
            adj_l_ncs.append((id2ind[b[1]],id2ind[b[2]]))
    return Sample(sequences, node_type, adj_l_ast, adj_l_cfg, adj_l_cdg, adj_l_ncs, str(js_file))


def gen(dir, fn_label, test=0.4, save=True) -> Tuple[Dataset, Dataset, Dataset]:

    if save:
        cache = os.path.join(dir, 'all.json.gz')
        print(cache)
        if os.path.exists(cache):
            with gzip.open(cache, 'rb') as rf:
                return pickle.load(rf)

    def _gen_files():
        all_fs = list(Path(dir).glob('**/*.asm.json')) if not 'devign' in dir else list(Path(dir).glob('**/*.json'))
        for f in tqdm(all_fs):
            c0, c1 = fn_label(str(f))
            yield json2sample(f), c0, c1

    ls = [s for s in _gen_files() if s]
    x, cwe, gob = list(zip(*ls))
    
    empty = [i for (i,j) in enumerate(x) if len(j.sequences) == 0]
    x = [j for (i,j) in enumerate(x) if i not in empty]
    cwe = [j for (i,j) in enumerate(cwe) if i not in empty]
    gob = [j for (i,j) in enumerate(gob) if i not in empty]
    
    
    ind2cwe = sorted(set(cwe))
    cwe2ind = {c: i for i, c in enumerate(ind2cwe)}
    
    cwe = [cwe2ind[c] for c in cwe]

    gob = np.array(gob)  # good or bad
    cwe = np.array(cwe)  # cwe code

    print('# label gob')
    print(pd.Series(gob).astype('category').describe())
    print('# label cwe')
    print(pd.Series(cwe).astype('category').describe())
    print('# block count')
    print(pd.Series([len(a.sequences) for a in x]).describe())
    print('# seq len')
    print(pd.Series([len(b)
                     for a in x for b in a.sequences]).describe())


    train, valid, test = train_valid_test_split(x, cwe, gob)
    ds = Dataset(*train, ind2cwe), Dataset(
        *valid, ind2cwe), Dataset(*test, ind2cwe)
    if save:
        with gzip.open(cache, 'wb') as wf:
            pickle.dump(ds, wf)
    return ds





def gen_devign_source():
    def label(js_file: str):
        js_file = os.path.basename(js_file)
        gob = 0 if 'good' in js_file else 1
        cwe = 121 if 'good' in js_file else 119
        return cwe, gob
    return gen(path_devign_source, fn_label=label)

if __name__ == "__main__":
    gen_devign_source()
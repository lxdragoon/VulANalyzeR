import gzip
import json
import pathlib
from argparse import ArgumentParser
from random import seed, shuffle
import os
import re

import sentencepiece as spm

# seed(0)

oprs_filter = (
    'loc_', 'sub_', 'arg_', 'var_', 'unk_',
    'word_', 'off_', 'locret_', 'flt_', 'dbl_', 'param_', 'local_')

TOKEN_PAD = '[PAD]'  # Token for padding
TOKEN_UNK = '[UNK]'  # Token for unknown words
TOKEN_CLS = '[CLS]'  # Token for classification
TOKEN_SEP = '[SEP]'  # Token for separation
TOKEN_MASK = '[MASK]'  # Token for masking

ID_PAD = 0
ID_UNK = 1
ID_CLS = 2
ID_SEP = 3
ID_MASK = 4

base_dict = {
    TOKEN_PAD: ID_PAD,
    TOKEN_UNK: ID_UNK,
    TOKEN_CLS: ID_CLS,
    TOKEN_SEP: ID_SEP,
    TOKEN_MASK: ID_MASK,
}


base_dict_rev = {
    ID_PAD: TOKEN_PAD,
    ID_UNK: TOKEN_UNK,
    ID_CLS: TOKEN_CLS,
    ID_SEP: TOKEN_SEP,
    ID_MASK: TOKEN_MASK,
}
supported_types = ['metapc', 'ppc']

tokenizer_foler = 'models/tokenizers/'
default_bin_home = '/home/shared-data/malbin/bins/'


def ins2seq(i):
    tkns = []
    for tkn in [i['mne']] + i['oprs']:
        for p in re.split(r'([+\-*\\\[\]:()\s@?_$])', tkn.lower()):
            if not p.startswith(oprs_filter) and len(p) > 0:
                tkns.append(p)
    return tkns


def blk2seq(b, sp: spm.SentencePieceProcessor = None, flatten=False):
    vals = [' '.join(ins2seq(i)) for i in b['ins']]
    if sp:
        vals = [sp.EncodeAsIds(v) for v in vals]
    if flatten:
        vals = [x for v in vals for x in v]
    return vals


def get_files(bins_path):
    return list(
        pathlib.Path(bins_path).glob('*.asm.json.gz'))


def get_tokenizer_lg(bins_path=default_bin_home):
    return get_tokenizer_xl(bins_path)

def get_tokenizer_xl(bins_path=default_bin_home, vocab_size = 100000):
    return get_tokenizer(bins_path, num_ins=10000000, vocab_size=vocab_size, re_train=False)


def get_tokenizer_md(bins_path=default_bin_home , vocab_size=20000):
    return get_tokenizer(bins_path, num_ins=1000000, vocab_size=vocab_size, re_train=False)


def get_tokenizer_sm(bins_path=default_bin_home, vocab_size=10000):
    return get_tokenizer(bins_path, num_ins=800000, vocab_size=vocab_size, re_train=False)


def get_tokenizer(bins_path, num_ins=10000, vocab_size=30000, re_train=False):

    folder = tokenizer_foler
    if not os.path.exists(folder):
        os.makedirs(folder)

    saved = 'tokenizer_{}_{}.sp'.format(
        num_ins, vocab_size)
    saved = os.path.join(folder, saved)
    model = saved + '.model'
    trained = False

    if not os.path.exists(model) or re_train:

        files = get_files(bins_path)
        shuffle(files)

        print('total', len(files), 'files at', bins_path)
        unknown_types = set()

        def generator():
            count = 0
            for file in files:
                with gzip.open(
                        file, 'rt',
                        encoding='utf-8') as zipfile:
                    obj = json.load(zipfile)
                    arch = obj['bin']['architecture']
                    if arch not in supported_types:
                        if arch not in unknown_types:
                            unknown_types.add(arch)
                            print('unknown', arch)
                        continue
                    blocks = obj['blocks']
                    for b in blocks:
                        if len(b['ins']) > 3:
                            values = blk2seq(b)
                            for v in values:
                                if len(v) > 0:
                                    count += 1
                                    if count >= num_ins:
                                        return
                                    yield v.lower()

        spm.SentencePieceTrainer.Train(
            sentence_iterator=generator(),
            model_prefix=saved,
            vocab_size=vocab_size,
            hard_vocab_limit=False,
            pad_piece=TOKEN_PAD,
            pad_id=ID_PAD,
            unk_piece=TOKEN_UNK,
            unk_id=ID_UNK,
            unk_surface=TOKEN_UNK,
            eos_id=-1,
            bos_id=-1,
            user_defined_symbols=[TOKEN_CLS, TOKEN_SEP, TOKEN_MASK])
        trained = True

    sp = spm.SentencePieceProcessor()
    sp.Load(model)
    if trained:
        print('actual vocab size', sp.vocab_size())
        print('uknown file types', unknown_types)

    setattr(sp, 'tokenize_blk', lambda s, b: [i['mne'] + ' ' + ' '.join(
        i['oprs']) for i in b['ins']])

    return sp


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument(
        '-p', '--json_path', type=str,
        default=default_bin_home
    )
    parser.add_argument(
        '-n', '--num_ins', type=int, default=10000)
    parser.add_argument(
        '-v', '--vocab_size', type=int, default=30000)
    flags = parser.parse_args()
    get_tokenizer(
        flags.json_path, flags.num_ins, flags.vocab_size,
        re_train=True)

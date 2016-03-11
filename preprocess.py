#!/usr/bin/env python

"""Language modeling preprocessing
"""

import numpy as np
import h5py
import argparse
import sys
import re
import codecs

# Your preprocessing, features construction, and word2vec code.
def construct_word2idx(wd):
    word2idx = {}
    with open(wd) as f:
        for i, line in enumerate(f):
            cword = line.split()
            word2idx[cword[1]] = cword[0]
    return word2idx     

def transform_sent(sent, word2idx, dwin):
    tsent = [word2idx['<s>']] * dwin
    for s in sent.split():
        try:
            tsent.append(word2idx[s])
        except:
            tsent.append(word2idx['<unk>'])
  
    tsent.append(word2idx['</s>'])
    return tsent

def convert_data(filename, word2idx, dwin):
    features = []
    lbl = []
    with open(filename) as f:
        for i, line in enumerate(f):
            sent = transform_sent(line, word2idx, dwin)
            for c in range(len(sent)-dwin):
                features.append(sent[c:dwin+c])
                lbl.append([sent[dwin+c]])
    return np.array(features, dtype=np.int32), np.array(lbl, dtype=np.int32)

def transform_test_sent(sent, word2idx, dwin):  
    split = sent.split()[:-1]
    tsent = [word2idx['<s>']] * max(0, dwin-len(split))
    #only store dwin last words
    if len(split) > dwin:
        split = split[-dwin:]
    for s in split:
        try:
            tsent.append(word2idx[s])
        except:
            tsent.append(word2idx['<unk>'])
    return tsent

def convert_test(filename, word2idx, dwin):
    features = []
    lbl_set = []
    with open(filename) as f:
        for i, line in enumerate(f):
            if (line[0] == "Q"):
                sent = transform_sent(line[2:], word2idx, dwin)

                lbl_set.append(sent)
            else:
                sent = transform_test_sent(line[2:], word2idx, dwin)
                features.append(sent)
    features = np.array(features, dtype=np.int32)
    lbl_set = np.array(lbl_set, dtype=np.int32)
    return features, lbl_set

def convert_valid_results(filename, word2idx, dwin):
    lbl = []
    with open(filename) as f:
        for i, line in enumerate(f):
            if i>0:
                r = line[:-1].split(',')[1:].index('1')+1

                lbl.append(r)
    return np.array(lbl, dtype=np.int32)

FILE_PATHS = {"PTB": ("data/train.txt",
                      "data/valid.txt",
                      "data/valid_blanks.txt",
                      "data/valid_kaggle.txt",
                      "data/test_blanks.txt",
                      "data/words.dict"),
              "PTB1000": ("data/train.1000.txt",
                      "data/valid.1000.txt",
                      "data/valid_blanks.txt",
                      "data/valid_kaggle.txt",
                      "data/test_blanks.txt",
                      "data/words.dict")}
args = {}


def main(arguments):
    global args
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('dataset', help="Data set",
                        type=str)
    parser.add_argument('dwin', help="Context Size",
                        type=int)
    args = parser.parse_args(arguments)
    dataset = args.dataset
    dwin = args.dwin-1
    train, valid, valid_blanks, valid_out, test, word_dict = FILE_PATHS[dataset]

    #code here
    word2idx = construct_word2idx(word_dict)

    train_input, train_output = convert_data(train, word2idx, dwin)
    #valid_input, valid_output = convert_data(valid, word2idx, dwin)
    valid_blanks_input, valid_blanks_set = convert_test(valid_blanks, word2idx, dwin)
    valid_out = convert_valid_results(valid_out, word2idx, dwin)
    test_input, test_set = convert_test(test, word2idx, dwin)

    C = len(word2idx)
    V = len(train_input)
    print(C, "Classes")
    print(V, "Training Sequences")

    filename = args.dataset + "-win-"+str(dwin+1) + '.hdf5'
    with h5py.File(filename, "w") as f:
        f['train_input'] = train_input
        f['train_output'] = train_output
        if valid:
            #f['valid_input'] = valid_input
            f['valid_output'] = valid_out
            f['valid_blanks_input'] = valid_blanks_input
            f['valid_blanks_set'] = valid_blanks_set
        if test:
            f['test_input'] = test_input
            f['test_set'] = test_set
        f['nclasses'] = np.array([C], dtype=np.int32)
        f['nfeatures'] = np.array([V], dtype=np.int32)
        f['dwin'] = np.array([dwin], dtype=np.int32)

if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))

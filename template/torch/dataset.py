# -*- coding: utf-8 -*-
"""encoder-decoder model implemented by pytorch"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
#from torchtext.data import Field, BucketIterator
import torch.utils.data.dataloader as dataloader
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import math
import time
import random
import sys
#sys.path.append('..')
import nlc_preprocess as nlc_preprocess
from nlc_preprocess import get_tokenizer
import numpy as np
import logging
from entmax import sparsemax
from entmax.losses import SparsemaxLoss

#device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

''' cuda multiprocessing'''
#from multiprocessing import set_start_method
#try:
    #set_start_method('spawn')
#except RuntimeError:
    #pass

PAD_ID = 0
SOS_ID = 1
EOS_ID = 2
UNK_ID = 3
_START_VOCAB = [b"<pad>", b"<sos>", b"<eos>", b"<unk>"]



class Corpus(): #randomly split training and testing data
    def __init__(self, corpus_dict):  # corpus = [{barcode: {error_ratio: xx, book: [[],[],[]...], page_num:xxx}}]
        self.corpus = corpus_dict
        self.barcodes = list(self.corpus.keys())
        #print(self.barcodes)

    def pages_from_id(self, barcode):
        #print(barcode)
        #print(self.corpus[barcode].keys())
        pages = self.corpus[barcode][b'book']
        return pages

    def get_pairs_flat(self, barcode, ratio=0.8):
        pages = self.pages_from_id(barcode)
        #print('page number: {}'.format(len(pages)))
        train_number = random.sample(range(len(pages)), int(len(pages)*ratio))
        test_number = [p for p in range(len(pages)) if p not in train_number]
        train, test = [],[]
        for p in train_number:
            train += pages[p]
        for p in test_number:
            test += pages[p]
        return train, test

    def get_random_ids(self, number):
        return random.sample(self.barcodes, number)

    def divide_to_groups(self, number=4):  # number books in one group
        barcodes = self.barcodes
        random.shuffle(barcodes)
        #print(barcodes)
        groups = []
        while barcodes:
            groups.append(barcodes[:number])
            barcodes = barcodes[number:]
            #print(barcodes)
        
        return groups

    def get_pairs_by_group(self, number=4, ratio=0.8):  # number books in one group
        self.group_ids = self.divide_to_groups(number)  #remember the random book id
        print('random group book ids: {}'.format(self.group_ids))
        Train_all = []
        Test_all = []
        for group in self.group_ids:
            Train = []
            Test = []
            for barcode in group:
                train, test = self.get_pairs_flat(barcode, ratio)
                Train += train
                Test += test

            Train_all.append(Train)
            Test_all.append(Test)

        return Train_all, Test_all



class LoadData():
    def __init__(self,  batch, data_path, target_path, vocab_path, tokenizer, max_vocab_size):
    #def __init__(self,  batch, tokenizer, max_vocab_size):
        self.batch = batch
        self.max_vocab_size = max_vocab_size
        self.data_path = data_path
        self.target_path = target_path
        self.vocab_path = vocab_path
        self.tokenizer = get_tokenizer(tokenizer)
    
        self.read_data()
        self.train_valid_split()

    
    def read_data(self):  # read training corpus from pickle file
        print('get dataset from {}'.format(self.data_path))
        data_x, data_y, max_length, vocab_size, vocab, vocab_reverse = nlc_preprocess.prepare_nlc_data(
            data_path=self.data_path, target_path=self.target_path, vocab_path=self.vocab_path,
            tokenizer=get_tokenizer(self.tokenizer))
        self.vocab = vocab
        self.vocab_reverse = vocab_reverse
        self.max_length = max_length
        self.data = [(x, y) for (x, y) in zip(data_x, data_y)]
        self.custom = CustomData(self.data, self.max_length)
    
    
    def build_vocab_on_the_fly(self, train_corpus): #read training instances directly
        vocab = {}
        ocr = [p[0] for p in train_corpus]
        truth = [p[1] for p in train_corpus]
        for sent in ocr+truth:
            sent = sent.decode('utf8', 'ignore')
            tokens = self.tokenizer(sent)
            for w in tokens:
                if w in vocab:
                    vocab[w] += 1
                else:
                    vocab[w] = 1
        vocab_list = _START_VOCAB + sorted(vocab, key=vocab.get, reverse=True)
        if len(vocab_list) > self.max_vocab_size:
            vocab_list = vocab_list[:self.max_vocab_size]

        self.vocab = dict([(y, x) for (x, y) in enumerate(vocab_list)])
        self.vocab_reverse = dict([(x, y) for (x, y) in enumerate(vocab_list)])


    def tokens_to_ids(self, sentence):
        sentence = sentence.decode('utf8', 'ignore')
        tokens = self.tokenizer(sentence)
        return [SOS_ID] + [self.vocab.get(w, UNK_ID) for w in tokens] + [EOS_ID]  # <sos>, <eos>


    def corpus_to_ids(self, corpus):  # corpus: list of tuples
        data = []
        lengths = []
        for (inp, out) in corpus:
            inp_id = self.tokens_to_ids(inp)
            out_id = self.tokens_to_ids(out)
            lengths.append(len(inp_id))
            lengths.append(len(out_id))
            data.append((inp_id, out_id))

        #self.max_length = max(lengths)
        #self.data = data
        return data, max(lengths)

    def train_valid_split(self, ratio=0.2):
        print('split training and validation dataset.')
        #index = int(len(self.data)*(1-ratio))
        #random.shuffle(self.data)
        index = int(len(self.data) * ratio)
        self.train = self.data[index:]
        self.valid = self.data[:index]
        print('train: {}'.format(len(self.train)))
        print('valid: {}'.format(len(self.valid)))
        self.train = self.custom_data(self.train)
        self.valid = self.custom_data(self.valid)
        self.test = self.custom_data(self.valid)

    def custom_data(self, instances):
        custom = CustomData(instances, self.max_length)
        return custom
    
    def prepare_corpus(self, Train, Test):
        self.build_vocab_on_the_fly(Train)
        self.data, self.max_length = self.corpus_to_ids(Train)
        self.test, _ = self.corpus_to_ids(Test)
        self.train_valid_split()
        # convert to torch dataset object
        self.train = self.custom_data(self.train)
        self.valid = self.custom_data(self.valid)
        self.test = self.custom_data(self.test)

    
    def prepare_other_corpus(self, Test):  # corpus to ids for other test data using existed vocab
        test_ids, _ = self.corpus_to_ids(Test)
        test_custom = self.custom_data(test_ids)
        return test_custom



class CustomData(Dataset):  # pytorch dataset class
    def __init__(self, instances, maxlen):
        self.maxlen = maxlen
        self.padded_data = [ (self.pad_data(s[0]), self.pad_data(s[1])) for s in instances]

    def __len__(self):
        
        return len(self.padded_data)

    def __getitem__(self, idx):
        x = self.padded_data[idx][0]
        #x_len = self.padded_data[idx][1]
        y = self.padded_data[idx][1]
        #y_len = self.padded_data[idx][3]
        return x, y

    def pad_data(self, s):
        padded = np.zeros((self.maxlen,), dtype=np.int64)
        if len(s) > self.maxlen: padded[:] = s[:self.maxlen]
        else: padded[:len(s)] = s
        return padded




if __name__ == '__main__':
    print('test')

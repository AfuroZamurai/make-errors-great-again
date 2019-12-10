## preprocess data for DL, character-spliting and so on
## created by Lijun, 21. Aug

import time
import os
import pickle
import random
import numpy as np
import nltk
from nltk import ngrams

_PAD = b"<pad>"
_SOS = b"<sos>"
_EOS = b"<eos>"
_UNK = b"<unk>"
_START_VOCAB = [_PAD, _SOS, _EOS, _UNK]

PAD_ID = 0
SOS_ID = 1
EOS_ID = 2
UNK_ID = 3


def get_data(path='pickle/GT_pairs_doris.pickle'):

    with open(path, 'rb')as f:
        sentence_pair = pickle.load(f)

    return sentence_pair


def char_tokenize(sentence):
    
    return list(sentence)


def ngram_tokenize(sentence, order=3):

    ## tokenize sentence to subwords sequence
    #words = sentence.split()
    grams = [''.join(n) for n in list(ngrams(sentence, order))]
    return grams

def ngram_tokenize_by_char(sentence, order=3):

    grams = [list(n) for n in list(ngrams(sentence, order))]
    return grams

def get_tokenizer(tokenizer):

    if tokenizer == 'char':
        return char_tokenize
    elif tokenizer == 'ngram':
        return ngram_tokenize
    else:
        return char_tokenize

def create_vocab_ngram():

    pass



def create_vocab(data_path, vocab_path, tokenizer, max_vocab_size=20000):

    vocab_dict = {}
    vocab_reverse = {}
    if not os.path.isfile(vocab_path):
        print('creating vocabulary %s from data %s' %(vocab_path, data_path))

        vocab = {}
        sentence_pair = get_data(path=data_path)
        sentent_ocr = [pair[0] for pair in sentence_pair]
        sentence_truth = [pair[1] for pair in sentence_pair]
        for sent in sentence_truth+sentent_ocr:
            sent = sent.decode('utf-8', 'ignore')     ## ascii convert to unicode
            tokens = tokenizer(sent)
            for w in tokens:
                if w in vocab:
                    vocab[w] += 1
                else:
                    vocab[w] = 1

        vocab_list = _START_VOCAB + sorted(vocab, key=vocab.get, reverse=True)
        print('vocabulary size: ' + str(len(vocab_list)))

        if len(vocab_list) > max_vocab_size:
            vocab_list = vocab_list[:max_vocab_size]

        vocab_dict = dict([(y, x) for (x,y) in enumerate(vocab_list)])

        with open(vocab_path,'wb')as f:
            pickle.dump(vocab_dict, f)

    else:
        with open(vocab_path, 'rb')as f:
            #vocab_dict = pickle.load(f)
            vocab_dict = pickle.load(f, encoding='bytes')

    ## get reverse_vocab
    #for x,y in vocab_dict.iteritems():
    for x,y in vocab_dict.items():
        vocab_reverse[y] = x

    return vocab_dict, vocab_reverse


def sentence_to_token_ids(sentence, vocab, tokenizer):

    tokens = tokenizer(sentence)
    return [1]+ [vocab.get(w.encode(), UNK_ID) for w in tokens]+[2]

def data_to_token_ids(data_path, target_path, vocab_path, tokenizer):

    sentence_pair_ids = []
    if not os.path.isfile(target_path):
        print('tokenizing data from %s' %data_path)
        vocab, vocab_reverse = create_vocab(data_path, vocab_path, tokenizer)
        vocab_size = len(vocab)
        sentence_pair = get_data(path=data_path)
        for pair in sentence_pair:
            ocr = sentence_to_token_ids(pair[0].decode('utf-8', 'ignore'), vocab, tokenizer)
            clean = sentence_to_token_ids(pair[1].decode('utf-8', 'ignore'), vocab, tokenizer)
            ocr = sentence_to_token_ids(pair[0], vocab, tokenizer)
            clean = sentence_to_token_ids(pair[1], vocab, tokenizer)
            sentence_pair_ids.append((ocr, clean))

        with open(target_path,'wb')as f:
            pickle.dump(sentence_pair_ids, f)

    else:
        with open(target_path, 'rb')as f:
            sentence_pair_ids = pickle.load(f, encoding='bytes')
            #sentence_pair_ids = pickle.load(f)

        vocab, vocab_reverse = create_vocab(data_path, vocab_path, tokenizer)
        #vocab_size = len(vocab)

    return sentence_pair_ids, vocab, vocab_reverse


def id_to_token(ids, vocab_path):

    with open(vocab_path, 'rb')as f:
        vocab = pickle.load(f)

    vocab_reverse = dict([(y,x) for (x,y) in vocab.items()])
    return ''.join(vocab_reverse[i] for i in ids)

def prepare_nlc_data(data_path, target_path, vocab_path, tokenizer):

    data_ids, vocab, vocab_reverse = data_to_token_ids(data_path, target_path,vocab_path,tokenizer)
    vocab_size = len(vocab)
    #max_length_x = max([len(pair[0]) for pair in data_ids])
    #max_length_y = max([len(pair[1]) for pair in data_ids])
    max_length = 60
    data_x = [pair[0] for pair in data_ids]
    data_y = [pair[1] for pair in data_ids]
    
    return data_x, data_y, max_length, vocab_size, vocab, vocab_reverse


def ngram_to_char_id(sentence, vocab, tokenizer=ngram_tokenize_by_char):

    grams = tokenizer(sentence)
    index = [[vocab.get(char, UNK_ID) for char in gram] for gram in grams] 
    head = [[1] + index[0][:2]]
    tail = [index[-1][-2:] + [2]]
    index = head + index + tail

    return index

def prepare_nlc_data_ngram_by_char(padded_data_x, padded_data_y, order=3):

    ## ngrams sequence from char tokenizer
    #print padded_data_x[:3]
    data_x_ngram = []
    data_y_ngram = []
    #seq_length = len(padded_data_x[0]) - order + 1
    for seq_x, seq_y in zip(padded_data_x, padded_data_y):
        x_ngram = [list(n) for n in list(ngrams(seq_x, order))] + [[0]*order]*(order-1)
        y_ngram = [list(n) for n in list(ngrams(seq_y, order))] + [[0]*order]*(order-1)
        data_x_ngram.append(np.array(x_ngram))
        data_y_ngram.append(np.array(y_ngram))

    #data_x_ngram = [[list(n) for n in list(ngrams(sequence, order))] for sequence in padded_data_x]
    #data_y_ngram = [[list(n) for n in list(ngrams(sequence, order))] for sequence in padded_data_y]

    return data_x_ngram, data_y_ngram



def prepare_nlc_data_others(data_path, target_path, vocab_path, tokenizer=char_tokenize):

    data_ids, vocab_size = data_to_token_ids(data_path,target_path,vocab_path,tokenizer)

    max_length_x = max([len(pair[0]) for pair in data_ids])
    max_length_y = max([len(pair[1]) for pair in data_ids])

    ## shuffle
    #shuffle(data_ids)
    length = len(data_ids)
    train_data = data_ids[:length/8*7]
    valid_data = data_ids[length/8*7: length/16*15]
    test_data = data_ids[length/16*15:]

    x_train_data = [pair[0] for pair in train_data]
    y_train_data = [pair[1] for pair in train_data]
    x_valid_data = [pair[0] for pair in valid_data]
    y_valid_data = [pair[1] for pair in valid_data]
    x_test_data = [pair[0] for pair in test_data]
    y_test_data = [pair[1] for pair in test_data]

    return x_train_data, y_train_data, x_valid_data, y_valid_data, x_test_data, y_test_data, vocab_path, vocab_size


def data_for_test(test_path, vocab):
    pass


def pair_iter(x_data, y_data, batch_size, num_layers, sort_and_shuffle=True):

    batches = []

    while True:
        if len(batches) == 0:
            refill(batches, x_data, y_data, batch_size, sort_and_shuffle=sort_and_shuffle)

        if len(batches) == 0:
            break
        
        x_tokens, y_tokens = batches.pop(0)
        x_padded, y_padded = padded(x_tokens, num_layers), padded(y_tokens, 1)

        source_tokens = np.array(x_padded).T
        source_mask = (source_tokens != PAD_ID).astype(np.int32)
        target_tokens = np.array(y_padded).T
        target_mask = (target_tokens != PAD_ID).astype(np.int32)

        yield(source_tokens, source_mask, target_tokens, target_mask)

    return


def refill(batches, x_data, y_data, batch_size, sort_and_shuffle=True):

    line_pairs = []
    #x_single, y_single = x_data.pop(0), y_data.pop(0)

    while x_data and y_data:
        x_single, y_single = x_data.pop(0), y_data.pop(0)
        if len(x_single) < FLAGs.max_seq_len and len(y_single) < FLAGS.max_seq_len:
            line_pairs.append((x_single, y_single))
        if len(linje_pairs) == batch_size *16:
            break

    if sort_and_shuffle:
        line_pairs = sorted(line_pairs, key=lambda e: len(e[0]))

    for batch_start in xrange(0, len(line_pairs), batch_size):
        x_batch, y_batch = zip(*line_pairs[batch_start: batch_start+batch_size])

        batches.append((x_batch, y_batch))

    if sort_and_shuffle:
        random.shuffle(batches)

    return

def padded(tokens, depth):

    maxlen = max(map(lambda x: len(x), tokens))
    align = pow(2, depth-1)
    padlen = maxlen + (align - maxlen) % align
    return map(lambda token_list: tokenlist + [PAD_ID]*(padlen - len(token_list)), tokens)


if __name__ == '__main__':

    data_path = 'pickle/GT_pairs_5_grams_doris.pickle'
    target_path = 'pickle/nlc_data_2_ids_5_grams.pickle'
    vocab_path = 'pickle/nlc_data_vocabulary_5_grams.pickle'

    prepare_nlc_data(data_path, target_path, vocab_path)

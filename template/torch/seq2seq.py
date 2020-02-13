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
import pickle
import sys
import os
#sys.path.append('..')
import statistic as statistic
import nlc_preprocess as nlc_preprocess
from nlc_preprocess import get_tokenizer
import numpy as np
import logging
try:
    import entmax
except:
    os.system('pip3 install entmax --user')

import entmax
from entmax import sparsemax
from entmax.losses import SparsemaxLoss
from Model import Encoder, Attention, Decoder, Seq2Seq
from dataset import LoadData, CustomData, Corpus
from argparse import ArgumentParser

from plot import plot

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

''' cuda multiprocessing'''
#from multiprocessing import set_start_method
#try:
    #set_start_method('spawn', True)
#except RuntimeError:
    #pass


def get_cmd_args():
    parser = ArgumentParser()
    parser.add_argument('-t', '--tokenizer', default='char', help='tokenizer')
    parser.add_argument('-d', '--dataset', help='[travel, century17]')
    parser.add_argument('-m', '--mode', help='train or test')
    parser.add_argument('-dir', '--directory')
    parser.add_argument('-enc_units', '--enc_units', type=int, default=256)
    parser.add_argument('-dec_units', '--dec_units', type=int, default=256)
    parser.add_argument('-emb', '--embedding_dim', type=int, default=128)
    parser.add_argument('-epoch', '--epoch', type=int, default=10)
    parser.add_argument('-batch', '--batch', type=int, default=64)
    parser.add_argument('-n_features', '--n_features', type=int, default=10000)
    parser.add_argument('-dropout', '--dropout', type=float, default=0.3)
    parser.add_argument('-clip', '--clip', type=float, default=0.1)
    parser.add_argument('-model_name', '--model_name', help='model name')
    parser.add_argument('-tf', '--tf', type=float, default=1.0, help='teacher forcing ratio')
    parser.add_argument('-sparse_max', '--sparse_max', type=int, default=0)
    args = parser.parse_args()

    return args



class Train:
    def __init__(self, inp_dim, out_dim, emb_dim, enc_hid, dec_hid, enc_drop, dec_drop, epoch, clip, sparse_max, tf, max_length, vocab, batch, device):
        self.inp_dim = inp_dim
        self.out_dim = out_dim
        self.emb_dim = emb_dim
        self.enc_hid = enc_hid
        self.dec_hid = dec_hid
        self.enc_drop = enc_drop
        self.dec_drop = dec_drop
        self.tf = tf
        self.max_length = max_length
        self.batch = batch
        self.device = device
        self.vocab = vocab

        self.attn = Attention(enc_hid, dec_hid, sparse_max=sparse_max)
        self.enc = Encoder(inp_dim, emb_dim, enc_hid, dec_hid, enc_drop)
        self.dec = Decoder(out_dim, emb_dim, enc_hid, dec_hid, dec_drop, self.attn)
        self.model = Seq2Seq(self.enc, self.dec, device).to(device)

        self.model.apply(self.init_weights)
        self.count_parameters()
        self.optimizer = optim.Adam(self.model.parameters())
        if sparse_max:
            self.criterion = SparsemaxLoss(ignore_index=0)
        else:
            self.criterion = nn.CrossEntropyLoss(ignore_index=0)  # pad_idx 0
        self.epoch = epoch
        self.clip = clip

    def init_weights(self, m):
        for name, param in m.named_parameters():
            if 'weight' in name:
                nn.init.normal_(param.data, mean=0, std=0.01)
            else:
                nn.init.constant_(param.data, 0)

    # model.apply(init_weights())
    def count_parameters(self):
        param_num = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print('The model has {} trainable parameters'.format(param_num))
        return param_num


    def train(self, train_data):
        print('train model...')
        self.model.train()
        epoch_loss = 0
        epoch_accu = 0
        iterator = iter(dataloader.DataLoader(train_data, batch_size=self.batch, num_workers=1, shuffle=True, pin_memory=True))  ## add pin_memory here to use GPU
        step = 0
        start_time = time.time()
        for batch in iterator:
            step += 1
            if step % 200 == 0:
                end_time = time.time()
                print('{0}/{1} steps in epoch done in {2:4f} seconds'.format(step, len(iterator), (end_time - start_time)))
                start_time = time.time()
            #print('batch: ', batch[0].shape)
            #print('batch: ', i)
            src = batch[0].permute(1,0)  #batch second
            #src_length = batch[1].permute(1,0) 
            trg = batch[1].permute(1,0)   # trg = [trg sent len, batch size]
            #trg_length = batch[3].permute(1,0)
            src = src.to(self.device)
            trg = trg.to(self.device)
            #src_length = src_length.to(self.device)
            #trg_length = trg_length.to(self.device)
            self.optimizer.zero_grad()
            output = self.model(src, trg, self.tf)   # output = [trg sent len, batch size, output dim]
            output = output[1:].view(-1, output.shape[-1])  # output = [(trg sent len - 1) * batch size, output dim]
            trg = trg[1:].view(-1)  # trg = [(trg sent len - 1) * batch size]
            predict = self.prob_to_index(output)
            accuracy = self.Accuracy(predict, trg)
            loss = self.criterion(output, trg)
            #print('loss: ', loss)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip)
            self.optimizer.step()
            epoch_loss += loss.item()
            epoch_accu += accuracy

        return epoch_loss / len(iterator), epoch_accu / len(iterator)


    def evaluate(self, valid_data):
        print('Evaluating model...')
        ws = self.vocab.get(' ') #get white space
        self.model.eval()
        epoch_loss = 0
        epoch_accu = 0
        epoch_wer = 0
        epoch_cer = 0
        epoch_wer_ocr = 0
        epoch_cer_ocr = 0
        #print('batch: ', self.batch)
        iterator = iter(dataloader.DataLoader(valid_data, batch_size=self.batch, num_workers=1, shuffle=True, pin_memory=True))  ## add pin_memory here to use GPU
        with torch.no_grad():
            for i, batch in enumerate(iterator):
                if i % 100 == 0:
                    print('Evaluated {0}/{1} samples'.format(i, len(iterator)))
                src = batch[0].permute(1,0).to(self.device)
                #src_length = batch[1].permute(1,0).to(self.device)
                trg = batch[1].permute(1,0).to(self.device)
                #trg_length = batch[3].permute(1,0).to(self.device)
                wer, cer = self.Edit_dist_in_batch(src, trg, ws)
                #print('ocr_wer: {} | ocr_cer: {} '.format(wer, cer))
                epoch_wer_ocr += wer
                epoch_cer_ocr += cer
                output = self.model(src, trg, 0)  # turn off teacher forcing
                #output = output[1:].view(-1, output.shape[-1]) # squeeze output
                output = output[1:]
                predict = self.prob_to_index(output)
                trg = trg[1:]
                # wer, cer in batch
                #print('predict: ', predict, predict.shape)
                #print('target: ', trg, trg.shape)
                
                wer, cer = self.Edit_dist_in_batch(predict, trg, ws)
                epoch_wer += wer
                epoch_cer += cer

                predict = predict.view(-1) # squeeze both
                trg = trg.view(-1)
                output = output.view(-1, output.shape[-1])
                accuracy = self.Accuracy(predict, trg)
                loss = self.criterion(output, trg)
                epoch_loss += loss.item()
                epoch_accu += accuracy
        #print('Accuracy for a batch: ', epoch_accu)
        iter_length = len(iterator)
        return epoch_loss/iter_length, epoch_accu/iter_length, epoch_wer_ocr/iter_length, epoch_cer_ocr/iter_length, epoch_wer/iter_length, epoch_cer/iter_length


    def start_train(self, train_data, valid_data, model_dir):
        results = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
            'wer_ocr': [],
            'wer_after': [],
            'cer_ocr': [],
            'cer_after': []
        }
        best_valid_loss = float('inf')
        for epoch in range(self.epoch):
            start = time.time()
            #print('training numbers: ', len(train_iterator))
            print('Starting epoch {0}'.format(epoch))
            
            train_loss, train_accuracy = self.train(train_data)
            valid_loss, valid_accuracy, valid_wer_ocr, valid_cer_ocr, valid_wer, valid_cer = self.evaluate(valid_data)
            end = time.time()
            mins = int((end-start)/60)
            if valid_loss < best_valid_loss:
                best_valid_loss = valid_loss
                torch.save(self.model.state_dict(), model_dir)

            results['train_loss'].append(train_loss)
            results['train_acc'].append(train_loss)
            results['val_loss'].append(train_loss)
            results['val_acc'].append(train_loss)
            results['wer_ocr'].append(valid_wer_ocr)
            results['wer_after'].append(valid_wer)
            results['cer_ocr'].append(valid_cer_ocr)
            results['cer_after'].append(valid_cer)

            print('Epoch: {}  | Time: {} m'.format(epoch+1, mins))
            print('\tTrain Loss: {} | Train Accuracy: {}'.format(train_loss, train_accuracy))
            print('\tVal. Loss: {} |  Val Accuracy: {}'.format(valid_loss, valid_accuracy))
            print('\tVal. WER_OCR: {} |  Val CER_OCR: {}'.format(valid_wer_ocr, valid_cer_ocr))
            print('\tVal. WER_After: {} |  Val CER_After: {}'.format(valid_wer, valid_cer))

            logging.info('Epoch: {}  | Time: {} m'.format(epoch+1, mins))
            logging.info('\tTrain Loss: {} | Train Accuracy: {}'.format(train_loss, train_accuracy))
            logging.info('\tVal. Loss: {} |  Val Accuracy: {}'.format(valid_loss, valid_accuracy))
            logging.info('\tVal. WER_OCR: {} |  Val CER_OCR: {}'.format(valid_wer_ocr, valid_cer_ocr))
            logging.info('\tVal. WER_After: {} |  Val CER_After: {}\n'.format(valid_wer, valid_cer))

        return results

    def load_model(self, model_dir):
        print('loading model from ', model_dir)
        self.model.load_state_dict(torch.load(model_dir))  # not sure if it needs return value


    def prob_to_index(self, out):
        _, predict = torch.max(out, -1)
        #predict = predict.view(-1)
        return predict

    def test(self, Data, test, model_dir):
        self.load_model(model_dir)
        #loss, accuracy = self.evaluate(test_data)
        return self.translate(Data, test)
        #print('Evaluate loss: {}, accuracy: {}'.format(loss, accuracy))

    def test_in_batch(self, test_data, model_dir):
        self.load_model(model_dir)
        loss, accuracy, wer_ocr, cer_ocr, wer_after, cer_after = self.evaluate(test_data)
        print('loss: {} | accuracy: {} | wer_ocr: {} | cer_ocr: {} | wer_after: {} | cer_after: {}'.format(loss, accuracy, wer_ocr, cer_ocr, wer_after, cer_after))

        
    def Accuracy(self, pred, truth):  ## both are squeezed
        num_correct = (pred == truth).sum().item()  # here the prediction after 2 (<eos>) is still 2.
        return float(num_correct)/len(truth.nonzero())
    

    def Edit_dist_in_batch(self, pred_batch, truth_batch, ws):  ## both are in batch, shape: [length, batch]
        #numpy array
        WER = 0
        CER = 0
        pred_batch = pred_batch.cpu().data.numpy()  ## convert GPU tensor to cpu
        truth_batch = truth_batch.cpu().data.numpy()
        #print(pred_batch.shape, truth_batch.shape)
        batch = truth_batch.shape[1]
        for i in range(batch):
            pred = pred_batch[:, i]
            truth = truth_batch[:, i]  ## todo: clean pad and eos, 0 and 2
            #print('rec: ', pred, 'ref: ', truth)
            #print('predict: ', pred, pred.shape)
            #print('Truth: ', truth)
            pred = self.remove_func_id(pred)
            truth = self.remove_func_id(truth)
            #print(pred, pred.shape)
            pred = list(pred)
            truth = list(truth)
            WER += statistic.word_error_rate(pred, truth, ws)  #todo: get ws
            CER += statistic.char_error_rate(pred, truth)
        return WER/self.batch, CER/self.batch

    
    def remove_func_id(self, arr): # remove the functional index e.g., <sos>, <eos>,<pad> 
        #exclude = np.concatenate((np.where(arr == 0)[0], np.where(arr == 1)[0]))
        mask_id = np.ones(arr.shape, bool)
        #eos_id = arr.shape[0]
        exclude_eos = np.where(arr == 2)[0]  # find the start of <eos>
        if exclude_eos.any():
            eos_id = exclude_eos[0]
            #mask_id[exclude] = False
            mask_id[eos_id: ] = False
        arr_clean = arr[mask_id]
        #print('remove func id: ', arr_clean)
        return arr_clean

    def translate(self, Data, sentences):  #generate correction for sent
        # char to index, dimension: [length, batch]
        #Data = Data.to(self.device)
        sent_ret = []
        start_time = time.time()
        accum_time = 0
        for i, sent in enumerate(sentences):
            if i % 1000 == 0 and i > 0:
                end_time = time.time()
                needed = (end_time - start_time)
                accum_time += needed
                remaining = (len(sentences) - i) * accum_time / i
                print('{0}/{1} sentences translated in {2:.1f} seconds, {3:.2f} minutes remaining'.
                      format(i, len(sentences), needed, (remaining / 60)))
                start_time = time.time()

            sent_out = ''
            try:
                encoder_input = nlc_preprocess.sentence_to_token_ids(sent, Data.vocab, get_tokenizer(Data.tokenizer))
                encoder_input = torch.from_numpy(Data.custom.pad_data(encoder_input)).view(-1, 1).to(self.device) ## todo: fix padding
            except (AttributeError, TypeError) as e:
                encoder_input = torch.from_numpy(sent[0]).view(-1, 1).to(self.device)
            # print('input: ', encoder_input)
            decoder_input = np.zeros(shape=( self.max_length, 1), dtype=np.int64)
            decoder_input[0, :] = nlc_preprocess.SOS_ID
            decoder_input = torch.from_numpy(decoder_input).to(self.device)
            #print('input shape: {}, output shape: {}'.format(encoder_input.shape, decoder_input.shape))
            output = self.model(encoder_input, decoder_input, 0)  ## turn off teacher forcing
            #_, output = torch.max(output, -1)
            output = self.prob_to_index(output).tolist()
            #print('output: ', output)
            for i in output[1:]:
                if i[0] == 2:
                    break

                v = Data.vocab_reverse[i[0]]
                try:
                    sent_out += v.decode('ISO-8859-1')
                except AttributeError:
                    sent_out += v

            expected = ''
            for i in sent[1][1:]:
                if i == 2:
                    break

                v = Data.vocab_reverse[i]
                try:
                    expected += v.decode('ISO-8859-1')
                except AttributeError:
                    expected += v

            #print('Expected sentence: ', expected)
            #print('Predicted sentence: ', sent_out)
            sent_ret.append((expected,sent_out))
        return sent_ret

    def translate_sent(self, Data, sent):
        # char to index, dimension: [length, batch]
        # Data = Data.to(self.device)
        sent_out = ''
        encoder_input = nlc_preprocess.sentence_to_token_ids(sent, Data.vocab, get_tokenizer(Data.tokenizer))
        encoder_input = torch.from_numpy(Data.custom.pad_data(encoder_input)).view(-1, 1).to(
            self.device)  ## todo: fix padding
        # print('input: ', encoder_input)
        decoder_input = np.zeros(shape=(self.max_length, 1), dtype=np.int64)
        decoder_input[0, :] = nlc_preprocess.SOS_ID
        decoder_input = torch.from_numpy(decoder_input).to(self.device)
        # print('input shape: {}, output shape: {}'.format(encoder_input.shape, decoder_input.shape))
        output = self.model(encoder_input, decoder_input, 0)  ## turn off teacher forcing
        # _, output = torch.max(output, -1)
        output = self.prob_to_index(output).tolist()
        # print('output: ', output)
        for i in output[1:]:
            if i[0] == 2:
                break

            v = Data.vocab_reverse[i[0]]
            try:
                sent_out += v.decode('ISO-8859-1')
            except AttributeError:
                sent_out += v

        print('Correct sentence: ', sent_out)
        return sent_out


def main():
    args = get_cmd_args()
    
    args.data_path = args.directory + 'PKL/{}_train_clean.pickle'.format(args.dataset)
    args.target_path = args.directory + 'PKL/{}_clean_data_indexs.pickle'.format(args.dataset)
    args.vocab_path = args.directory + 'PKL/{}_clean_vocab.pickle'.format(args.dataset)
    args.model_dir = args.directory + 'Model/char/torch/{}.model'.format(args.model_name)
    log_file = args.directory + 'log/' + args.model_name + '_debug.log'
    logging.basicConfig(filename=log_file, filemode='w', level=logging.DEBUG)

    # test
    
    loaddata = LoadData(args.batch, args.data_path, args.target_path, args.vocab_path, args.tokenizer, args.n_features)
    args.inp_dim = args.out_dim = len(loaddata.vocab)
    args.max_len = loaddata.max_length
    args.vocab = loaddata.vocab
    task = Train(args.inp_dim, args.out_dim, args.embedding_dim, args.enc_units, args.dec_units, args.dropout, args.dropout, args.epoch, args.clip, args.sparse_max, args.tf, args.max_len, args.vocab, args.batch, device)

    if args.mode == 'train':
        logging.info('start training...')
        results = task.start_train(loaddata.train, loaddata.valid, args.model_dir)

        for k, v in results.items():
            print('{0}: {1}'.format(k, v))
            logging.info('{0}: {1}'.format(k, v))

        extension = '_' + str(args.model_name)

        save_path = args.directory + 'results/'

        # plot accuracy
        plot('accuracy' + extension, 'epochs', 'accuracy', results['train_acc'], results['val_acc'],
             'train accuracy', 'validation accuracy', save_path=save_path)

        # plot loss
        plot('loss' + extension, 'epochs', 'loss', results['train_loss'], results['val_loss'],
             'train loss', 'validation loss', save_path=save_path)

        # plot wer
        plot('wer' + extension, 'epochs', 'wer', results['wer_ocr'], results['wer_after'],
             'wer ocr', 'val wer', save_path=save_path)

        # plot cer
        plot('cer' + extension, 'epochs', 'cer', results['cer_ocr'], results['cer_after'],
             'cer ocr', 'val cer', save_path=save_path)

    else:
        logging.info('start testing...')
        # sent_clean = 'Mohren plagen uns ohne aufhÃ¶rlich'
        # sent_res = task.translate_sent(loaddata, sent_clean)
        sent_out = task.test(loaddata, loaddata.valid, args.model_dir)

        output_file = args.directory + 'log/' + args.model_name + '_output.txt'
        print('Saving to {0}\n'.format(output_file))
        with open(output_file, 'w', encoding='utf-8') as f:
            for sent_pair in sent_out:
                f.write(sent_pair[0] + ',' + sent_pair[1] + '\n')


if __name__ == '__main__':
    main()






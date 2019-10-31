"""
Take the given dataset and bring in a form suited for our model
"""

import os
import pandas as pd
import numpy as np


class Preprocessor:
    def __init__(self, data_path):
        self.data_path = data_path
        self.input_texts = []
        self.target_texts = []
        self.input_characters = set()
        self.target_characters = set()
        self.vectorize()

    def vectorize(self):
        # Vectorize the data.
        with open(self.data_path, 'r', encoding='utf-8') as f:
            header = f.readline()
            lines = f.read().split('\n')
        for line in lines:
            try:
                input_text, target_text = line.split(',')
            except:
                print(line)
            # We use "tab" as the "start sequence" character
            # for the targets, and "\n" as "end sequence" character.
            target_text = '\t' + target_text + '\n'
            self.input_texts.append(input_text)
            self.target_texts.append(target_text)
            for char in input_text:
                if char not in self.input_characters:
                    self.input_characters.add(char)
            for char in target_text:
                if char not in self.target_characters:
                    self.target_characters.add(char)

    def to_csv(self, path):
        data = {
            'input': [],
            'output': []
        }
        cur = 'input'
        with open(path, encoding='ISO-8859-1') as f:
            for line in f:
                if line == '\n':
                    cur = 'input'
                    continue
                else:
                    stripped = line.rstrip('\n')
                    if stripped != '\n':
                        data[cur].append(stripped)
                    cur = 'output'

        df = pd.DataFrame(data)
        df.to_csv(path_or_buf=os.path.join(os.getcwd(), 'data', 'smaller.csv'), index=False, encoding='utf-8')

    def get_charwise_data(self):
        input_characters = sorted(list(self.input_characters))
        target_characters = sorted(list(self.target_characters))
        num_encoder_tokens = len(input_characters)
        num_decoder_tokens = len(target_characters)
        max_encoder_seq_length = max([len(txt) for txt in self.input_texts])
        max_decoder_seq_length = max([len(txt) for txt in self.target_texts])

        print('Number of samples:', len(self.input_texts))
        print('Number of unique input tokens:', num_encoder_tokens)
        print('Number of unique output tokens:', num_decoder_tokens)
        print('Max sequence length for inputs:', max_encoder_seq_length)
        print('Max sequence length for outputs:', max_decoder_seq_length)

        input_token_index = dict(
            [(char, i) for i, char in enumerate(input_characters)])
        target_token_index = dict(
            [(char, i) for i, char in enumerate(target_characters)])

        encoder_input_data = np.zeros(
            (len(self.input_texts), max_encoder_seq_length, num_encoder_tokens),
            dtype='float32')

        decoder_input_data = np.zeros(
            (len(self.input_texts), max_decoder_seq_length, num_decoder_tokens),
            dtype='float32')

        decoder_target_data = np.zeros(
            (len(self.input_texts), max_decoder_seq_length, num_decoder_tokens),
            dtype='float32')

        for i, (input_text, target_text) in enumerate(zip(self.input_texts, self.target_texts)):
            for t, char in enumerate(input_text):
                encoder_input_data[i, t, input_token_index[char]] = 1.
            encoder_input_data[i, t + 1:, input_token_index[' ']] = 1.
            for t, char in enumerate(target_text):
                # decoder_target_data is ahead of decoder_input_data by one timestep
                decoder_input_data[i, t, target_token_index[char]] = 1.
                if t > 0:
                    # decoder_target_data will be ahead by one timestep
                    # and will not include the start character.
                    decoder_target_data[i, t - 1, target_token_index[char]] = 1.
            decoder_input_data[i, t + 1:, target_token_index[' ']] = 1.
            decoder_target_data[i, t:, target_token_index[' ']] = 1.

        return encoder_input_data, decoder_input_data, decoder_target_data


if __name__ == '__main__':
    p = os.path.join(os.getcwd(), 'data', 'smallest.txt')
    processor = Preprocessor()
    processor.to_csv(p)

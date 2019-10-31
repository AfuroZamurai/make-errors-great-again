"""
Take the given dataset and bring in a form suited for our model
"""

import os
import pandas as pd


def to_csv(path):
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
                data[cur].append(line.rstrip('\n'))
                cur = 'output'

    df = pd.DataFrame(data)
    df.to_csv(path_or_buf=os.path.join(os.getcwd(), 'data', 'smaller.csv'), index=False, encoding='utf-8')


def get_charwise_encoder_input():
    return


def get_charwise_decoder_input():
    return


def get_charwise_decoder_target():
    return


if __name__ == '__main__':
    p = os.path.join(os.getcwd(), 'data', 'smaller.txt')
    to_csv(p)
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
    with open(path, encoding='utf-32') as f:
        for line in f:
            if line == '\n':
                cur = 'input'
                continue
            else:
                data[cur].append(line.rstrip('\n'))
                cur = 'output'

    df = pd.DataFrame(data)
    df.to_csv(path_or_buf=os.path.join(os.getcwd(), 'data', 'smallest.csv'), index=False, encoding='utf-32')


if __name__ == '__main__':
    p = os.path.join(os.getcwd(), 'data', 'smallest.txt')
    to_csv(p)
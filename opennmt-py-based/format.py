"""
Format the dataset so it conforms to the expectations of OpenNMT-py.
This means separate files for source and target sentences. In the file sentences are separated by newlines.
To recognize words they must be tokenized (which just means separated by a space).
"""

import os
import glob


def format(data_path, save_path, valid_size, test_size):
    files = glob.glob(data_path)

    source = []
    target = []

    for file in files:
        i = 0
        with open(file, encoding='ISO-8859-1') as f:
            for line in f:
                if i % 3 == 0:
                    source.append(line)
                elif i % 3 == 1:
                    target.append(line)
                i += 1

    print(len(source))
    print(len(target))

    source.reverse()
    target.reverse()

    with open(os.path.join(save_path, 'valid_clean_file.txt'), mode='w', encoding='utf-8') as f:
        for i in range(0, valid_size):
            if i == valid_size - 1:
                f.write(source[i].rstrip())
            else:
                f.write(source[i])

    with open(os.path.join(save_path, 'valid_error_file.txt'), mode='w', encoding='utf-8') as f:
        for i in range(0, valid_size):
            if i == valid_size - 1:
                f.write(target[i].rstrip())
            else:
                f.write(target[i])

    with open(os.path.join(save_path, 'test_clean_file.txt'), mode='w', encoding='utf-8') as f:
        for i in range(valid_size, valid_size + test_size):
            if i == valid_size + test_size - 1:
                f.write(source[i].rstrip())
            else:
                f.write(source[i])

    with open(os.path.join(save_path, 'test_error_file.txt'), mode='w', encoding='utf-8') as f:
        for i in range(valid_size, valid_size + test_size):
            if i == valid_size + test_size - 1:
                f.write(target[i].rstrip())
            else:
                f.write(target[i])

    with open(os.path.join(save_path, 'train_clean_file.txt'), mode='w', encoding='utf-8') as f:
        for i in range(valid_size + test_size, len(source)):
            if i == len(source) - 1:
                f.write(source[i].rstrip())
            else:
                f.write(source[i])

    with open(os.path.join(save_path, 'train_error_file.txt'), mode='w', encoding='utf-8') as f:
        for i in range(valid_size + test_size, len(target)):
            if i == len(target) - 1:
                f.write(target[i].rstrip())
            else:
                f.write(target[i])


if __name__ == '__main__':
    format('../data/AI_lab_data/*.txt', os.path.join(os.getcwd(), '../', 'data'), valid_size=5000, test_size=31276)

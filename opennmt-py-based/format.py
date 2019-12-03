"""
Format the dataset so it conforms to the expectations of OpenNMT-py.
This means separate files for source and target sentences. In the file sentences are separated by newlines.
To recognize words they must be tokenized (which just means separated by a space).
"""

import os
import glob


def transform_newspaper(data_path, save_path):
    newspaper = []
    with open(data_path, encoding='utf-8') as f:
        for line in f:
            newspaper.append(line)

    for i, line in enumerate(newspaper):
        newline = "\n".join([
            " ".join([c if c != " " else "<s>" for c in line])
        ])
        newspaper[i] = newline.rstrip() + '\n'

    with open(os.path.join(save_path, 'newspaper_clean_charwise.txt'), mode='w', encoding='utf-8') as f:
        for i in range(0, len(newspaper)):
            if i == len(newspaper) - 1:
                f.write(newspaper[i].rstrip())
            else:
                f.write(newspaper[i])


def reformat(data_path, save_path):
    translations = []
    with open(data_path, encoding='utf-8') as f:
        for line in f:
            translations.append(line)

    detokenized = []
    for sentence in translations:
        desplit = sentence.replace(' ', '')
        final = desplit.replace('<s>', ' ')
        detokenized.append(final)

    with open(save_path, mode='w', encoding='utf-8') as f:
        for sentence in detokenized:
            f.write(sentence)


def format(data_path, save_path, valid_size, test_size, charwise=False):
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

    if charwise:
        for i, line in enumerate(source):
            newline = "\n".join([
                " ".join([c if c != " " else "<s>" for c in line])
            ])
            source[i] = newline.rstrip() + '\n'

        for i, line in enumerate(target):
            newline = "\n".join([
                " ".join([c if c != " " else "<s>" for c in line])
            ])
            target[i] = newline.rstrip() + '\n'

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
    reformat('../data/results/newspaper_output_charwise.txt', '../data/results/detokenized_newspaper.txt')
    #format('../data/AI_lab_data/*.txt', os.path.join(os.getcwd(), '../', 'data'),
     #      valid_size=5000, test_size=31276, charwise=False)
    #transform_newspaper('../data/newspaper/newspaper_clean.txt', '../data/newspaper')

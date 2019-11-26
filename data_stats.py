"""
Compute statistics about the datasets.
"""


def load_dataset(path):
    dataset = []
    with open(path, 'r', encoding='utf-8')as f:
        for line in f.readlines():
                dataset.append(line)

    return dataset


def get_tokens(dataset):
    tokens = set()
    for sentence in dataset:
        t = [x.strip() for x in sentence.split(' ')]
        tokens.update(t)

    return tokens


def get_chars(dataset):
    chars = set()
    for sentence in dataset:
        t = [x for x in sentence]
        chars.update(t)

    return chars


def print_and_load_vocab(dataset, msg):
    vocab = get_tokens(dataset)
    print('{0} {1}'.format(msg, len(vocab)))
    return vocab


def print_and_load_chars(dataset, msg):
    chars = get_chars(dataset)
    print('{0} {1}'.format(msg, len(chars)))
    return chars


def print_intersection(s1, s2, msg):
    intersection = s1.intersection(s2)
    print('{0} {1}'.format(msg, len(intersection)))


if __name__ == '__main__':
    train_clean = load_dataset('data/train_clean_file.txt')
    train_error = load_dataset('data/train_error_file.txt')
    valid_clean = load_dataset('data/valid_clean_file.txt')
    valid_error = load_dataset('data/valid_error_file.txt')
    test_clean = load_dataset('data/test_clean_file.txt')
    test_error = load_dataset('data/test_error_file.txt')
    test_output = load_dataset('data/test_output.txt')
    newspaper_clean = load_dataset('data/newspaper/newspaper_clean.txt')
    newspaper_output = load_dataset('data/newspaper/newspaper_output.txt')

    train_clean_vocab = print_and_load_vocab(train_clean, 'train clean vocab size:')
    train_error_vocab = print_and_load_vocab(train_error, 'train error vocab size:')
    print_intersection(train_clean_vocab, train_error_vocab, 'train ocab intersection size:')

    newspaper_clean_vocab = print_and_load_vocab(newspaper_clean, 'newspaper clean vocab size:')
    print_intersection(train_clean_vocab, newspaper_clean_vocab, 'train and newspaper vocab intersection size:')

    train_clean_chars = print_and_load_chars(train_clean, 'train clean chars size:')
    train_error_chars = print_and_load_chars(train_error, 'train error chars size:')
    print_intersection(train_clean_chars, train_error_chars, 'train chars intersection size:')

    newspaper_clean_chars = print_and_load_chars(newspaper_clean, 'newspaper clean chars size:')
    print_intersection(train_clean_chars, newspaper_clean_chars, 'train and newspaper chars intersection size:')

    test_clean_vocab = print_and_load_vocab(test_clean, 'test clean vocab size:')
    test_error_vocab = print_and_load_vocab(test_error, 'test error vocab size:')
    print_intersection(test_clean_vocab, test_error_vocab, 'test vocab intersection size:')

    test_output_vocab = print_and_load_vocab(test_output, 'test output vocab size:')
    print_intersection(test_error_vocab, test_output_vocab, 'test error and output vocab intersection size:')
    print_intersection(test_clean_vocab, test_output_vocab, 'test clean and output vocab intersection size:')
    print_intersection(newspaper_clean_vocab, test_output_vocab, 'test output and newspaper vocab intersection size:')

    test_clean_chars = print_and_load_chars(test_clean, 'test clean chars size:')
    test_error_chars = print_and_load_chars(test_error, 'test error chars size:')
    print_intersection(test_clean_chars, test_error_chars, 'test chars intersection size:')

    test_output_chars = print_and_load_chars(test_output, 'test output chars size:')
    print_intersection(test_error_chars, test_output_chars, 'test error and output chars intersection size:')
    print_intersection(test_clean_chars, test_output_chars, 'test clean and output chars intersection size:')
    print_intersection(newspaper_clean_chars, test_output_chars, 'test output and newspaper chars intersection size:')

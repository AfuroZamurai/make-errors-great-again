from Levenshtein import distance


def compute_levenshtein(file1, file2):
    dist_min = 1000
    dist_max = 0
    dist_sum = 0
    max_pair = ()
    count = 0
    count_equals = 0
    for s1, s2 in zip(file1, file2):
        dist = distance(s1, s2)
        if dist < 8 and dist > 0:
            count += 1
            dist_sum += dist
            if dist < dist_min:
                dist_min = dist
            if dist > dist_max:
                dist_max = dist
                max_pair = (s1, s2)
        elif dist == 0:
            count_equals += 1

    dist_avg = dist_sum / count

    print('samples: {0}\ndist_sum: {1}\ndist_avg: {2}\n'
          'dist_min: {3}\ndist_max: {4}\nmax_pair: {5}\nequals: {6}'.format(
        count, dist_sum, dist_avg, dist_min, dist_max, max_pair, count_equals
    ))


if __name__ == '__main__':
    print('Loading test clean...')
    test_clean = []
    with open('data/test_clean_file.txt', 'r', encoding='utf-8') as f:
        for line in f.readlines():
            test_clean.append(line.rstrip())
    test_clean = test_clean[0:7920]

    print('Loading test error...')
    test_error = []
    with open('data/test_error_file.txt', 'r', encoding='utf-8') as f:
        for line in f.readlines():
            test_error.append(line.rstrip())
    test_error = test_error[0:7920]

    print('Loading our test error...')
    test_our = []
    with open('data/test_output.txt', 'r', encoding='utf-8') as f:
        for line in f.readlines():
            test_our.append(line.rstrip())
    test_our = test_our[0:7920]

    print('Loading newspaper clean...')
    newspaper_clean = []
    with open('data/newspaper/newspaper_clean.txt', 'r', encoding='utf-8') as f:
        for line in f.readlines():
            newspaper_clean.append(line.rstrip())

    print('Loading newspaper error...')
    newspaper_error = []
    with open('data/newspaper/newspaper_output.txt', 'r', encoding='utf-8') as f:
        for line in f.readlines():
            newspaper_error.append(line.rstrip())

    print('Loading charwise error...')
    charwise_error = []
    with open('data/results/detokenized_test.txt', 'r', encoding='utf-8') as f:
        for line in f.readlines():
            charwise_error.append(line.rstrip())

    print('Loading newspaper charwise error...')
    charwise_newspaper_error = []
    with open('data/results/detokenized_newspaper.txt', 'r', encoding='utf-8') as f:
        for line in f.readlines():
            charwise_newspaper_error.append(line.rstrip())

    print('Loading newspaper charwise error big...')
    charwise_newspaper_error_big = []
    with open('data/results/detokenized_newspaper_big.txt', 'r', encoding='utf-8') as f:
        for line in f.readlines():
            charwise_newspaper_error_big.append(line.rstrip())

    # compute distance for ground truth
    print('Computing Levenshtein for test clean/test error...')
    compute_levenshtein(test_clean, test_error)

    # compute distance for test translation to clean ground truth
    print('Computing Levenshtein for test clean/test our error...')
    compute_levenshtein(test_clean, test_our)

    # compute distance for test translation to error ground truth
    print('Computing Levenshtein for test error/test our error...')
    compute_levenshtein(test_our, test_error)

    # compute distance for newspaper translation to clean ground truth
    print('Computing Levenshtein for newspaper clean/newspaper error...')
    compute_levenshtein(newspaper_clean, newspaper_error)

    # compute distance for test error to charwise error
    print('Computing Levenshtein for test error/ charwise error...')
    compute_levenshtein(test_error, charwise_error)

    # compute distance for test clean to charwise error
    print('Computing Levenshtein for test clean/ charwise error...')
    compute_levenshtein(test_clean, charwise_error)

    # compute distance for newspaper translation to clean ground truth
    print('Computing Levenshtein for newspaper clean/newspaper charwise error...')
    compute_levenshtein(newspaper_clean, charwise_newspaper_error)

    # compute distance for newspaper translation big to clean ground truth
    print('Computing Levenshtein for newspaper clean/newspaper charwise error big...')
    compute_levenshtein(newspaper_clean, charwise_newspaper_error_big)

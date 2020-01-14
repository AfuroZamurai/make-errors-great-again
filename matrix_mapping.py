from collections import defaultdict
import Bio.pairwise2 as pw
from nltk import ngrams
from datasketch import MinHash, MinHashLSH


def map_ocr_to_clean(clean, ocr, threshhold=2):
    m = defaultdict(list)
    # Create an MinHashLSH index optimized for Jaccard threshold 0.5,
    # that accepts MinHash objects with 128 permutations functions
    # lsh = MinHashLSH(threshold=0.4, num_perm=128)
    i = 0
    for o, c in zip(ocr, clean):
        i += 1
        print('Pair {0}:\n'.format(i))
        ocr_words = o.split(' ')
        clean_words = c.split(' ')
        print('OCR: {0}\nClean: {1}'.format(o, c))
        # minhashes = {}
        for ow in ocr_words:
            for cw in clean_words:
                # minhash = MinHash(num_perm=128)
                # for d in ngrams(cw, 3):
                #     minhash.update(''.join(d).encode('utf-8'))
                # lsh.insert(ow, minhash)
                # minhashes[ow] = minhash

                print('score')
                score = pw.align.globalxs(ow, cw, -1, -.5, score_only=True)
                print('alignments')
                alignments = pw.align.globalxs(ow, cw, -1, -.5, score_only=False)
                print('ow: {0}, cw: {1}'.format(ow, cw))
                if abs(len(cw) - score) < threshhold:
                    print(len(cw))
                    print(alignments)

        # for i in range(len(minhashes.keys())):
        #     result = lsh.query(minhashes[i])
        #     print('Candidates with Jaccard similarity > 0.5 for input {0}: {1}'.format(i, result))

    return m


if __name__ == '__main__':
    data = {
        'input': [],
        'output': []
    }
    cur = 'output'
    with open('data/AI_lab_data/Z124117102_auto_clean.txt', mode='r', encoding='ISO-8859-1') as f:
        for line in f:
            if line == '\n':
                cur = 'output'
                continue
            else:
                stripped = line.rstrip('\n')
                if stripped != '\n':
                    u = str(stripped.encode(), 'utf-8')
                    data[cur].append(u)
                cur = 'input'

    matrix = map_ocr_to_clean(data['input'], data['output'])

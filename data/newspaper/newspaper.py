import pickle


if __name__ == '__main__':
    # open a file, where you stored the pickled data
    file = open('Zeitung_17_sents.pkl', 'rb')

    # dump information to that file
    data = pickle.load(file)

    # close the file
    file.close()

    dataset = []
    for article in data:
        for sentence in article:
            dataset.append(sentence)

    print(len(dataset))

    with open('newspaper_clean.txt', 'w', encoding='utf-8') as f:
        for i, sentence in enumerate(dataset):
            f.write(sentence)
            if i != len(dataset) - 1:
                f.write('\n')

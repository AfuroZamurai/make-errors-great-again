import pickle

# open a file, where you stored the pickled data
file = open('travel_train_clean.pickle', 'rb')

# dump information to that file
data = pickle.load(file, encoding='ISO-8859-1')

# close the file
file.close()

pairs = []
with open('newspaper_training.txt', 'r', encoding='utf-8') as f:
    for line in f.readlines():
        split = line.split('\t')
        pairs.append((split[0], split[1].strip()))

data.extend(pairs)

with open('training_data.pickle', 'wb') as f:
    pickle.dump(data, f)

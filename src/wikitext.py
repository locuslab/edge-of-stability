import torch
import os
from torch.utils.data.dataset import TensorDataset

PREPROCESSINGS = ["raw"]
device = torch.device('cuda')
bptt = 35

def get_batch(source, i,bptt=35):
    seq_len = min(bptt, len(source) - 1 - i)
    data = source[i:i+seq_len]
    target = source[i+1:i+1+seq_len]
    return data, target

def make_all_batches(dataset):
    X = []
    Y = []
    for batch, i in enumerate(range(0, dataset.size(0) - 1, bptt)):
        data, target = get_batch(dataset, i, bptt)
        X.append(data)
        Y.append(target)
    return TensorDataset(torch.cat(X, dim=0), torch.cat(Y, dim=0))

def load_wikitext_2(preprocessing: str):

    location = os.path.join(os.environ["DATASETS"], "wikitext_2_" + preprocessing)

    train = torch.load(f"{location}/train")
    test = torch.load(f"{location}/test")

    train = make_all_batches(train)
    test = make_all_batches(test)

    return train, test


def save_wikitext_2(preprocessing: str, train, val, test):
    # assert preprocessing in PREPROCESSINGS
    location = os.path.join(os.environ["DATASETS"], "wikitext_2_" + preprocessing)
    os.makedirs(location, exist_ok='True')

    torch.save(train, f"{location}/train")
    torch.save(val, f"{location}/test")
    torch.save(test, f"{location}/real_test")


class Dictionary(object):
    def __init__(self):
        self.word2idx = {}
        self.idx2word = []

    def add_word(self, word):
        if word not in self.word2idx:
            self.idx2word.append(word)
            self.word2idx[word] = len(self.idx2word) - 1
        return self.word2idx[word]

    def __len__(self):
        return len(self.idx2word)


class Corpus(object):
    def __init__(self, path):
        self.dictionary = Dictionary()
        self.train = self.tokenize(os.path.join(path, 'train.txt'))
        self.valid = self.tokenize(os.path.join(path, 'valid.txt'))
        self.test = self.tokenize(os.path.join(path, 'test.txt'))

    def tokenize(self, path):
        print(path)
        """Tokenizes a text file."""
        assert os.path.exists(path)
        # Add words to the dictionary
        with open(path, 'r', encoding="utf8") as f:
            for line in f:
                words = line.split() + ['<eos>']
                for word in words:
                    self.dictionary.add_word(word)

        # Tokenize file content
        with open(path, 'r', encoding="utf8") as f:
            idss = []
            for line in f:
                words = line.split() + ['<eos>']
                ids = []
                for word in words:
                    ids.append(self.dictionary.word2idx[word])
                idss.append(torch.tensor(ids).type(torch.int64))
            ids = torch.cat(idss)

        return ids


def batchify(data, bsz):
    nbatch = data.size(0) // bsz
    data = data.narrow(0, 0, nbatch * bsz)
    data = data.view(bsz, -1).t().contiguous()

    return data.to(device)


if __name__ == "__main__":
    corpus = Corpus(os.environ["DATASETS"] + "/wikitext_2_raw")

    batch_size = 100
    train = batchify(corpus.train, batch_size)
    val = batchify(corpus.valid, batch_size)
    real_test = batchify(corpus.test, batch_size)

    save_wikitext_2(f'raw-{batch_size}', train, val, real_test)

